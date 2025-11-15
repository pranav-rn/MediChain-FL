# medichain-fl/backend/fl_client/client.py
import base64
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from pathlib import Path
import sys
from typing import Optional
from PIL import Image  # Required for Hugging Face processor

# Add the parent directory (backend) to sys.path to import model.py
sys.path.append(str(Path(__file__).parent.parent))
from model import load_model, load_image_processor  # Import both functions
from utils.encryption import HEManager


class HuggingFaceImageFolder(Dataset):
    """
    A wrapper for ImageFolder to apply Hugging Face's image processor.
    """

    def __init__(self, root: str, image_processor):
        self.image_processor = image_processor
        # ImageFolder will load PIL images by default
        self.dataset = datasets.ImageFolder(root)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        processed_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed_inputs["pixel_values"].squeeze(0)
        return pixel_values, label


class PneumoniaClient(fl.client.NumPyClient):
    def __init__(
        self,
        hospital_id: str,
        data_path: str,
        model_name: str = "dima806/chest_xray_pneumonia_detection",
        device: str = "cpu",
    ):
        self.hospital_id = hospital_id
        self.device = device
        self.he_manager: Optional[HEManager] = None
        self.he_context_id: str | None = None

        # Load Hugging Face model
        self.model = load_model(model_name=model_name, freeze_encoder=True).to(device)

        # Load processor
        self.image_processor = load_image_processor(model_name=model_name)

        # Dataset
        self.trainset = HuggingFaceImageFolder(
            data_path, image_processor=self.image_processor
        )
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=8,  # Kept the reduced batch size
            shuffle=True,
            num_workers=0,  # IMPORTANT CHANGE: Set num_workers to 0 for stability on macOS/Docker
            pin_memory=False,  # IMPORTANT CHANGE: Set pin_memory to False when not using a GPU
        )

        # Loss + optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.001
        )

        print(
            f"[{self.hospital_id}] Initialized with {len(self.trainset)} samples. Device: {self.device}"
        )
        print(
            f"[{self.hospital_id}] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def get_parameters(self, config):
        """
        Returns only the model parameters that are trainable (unfrozen).
        """
        print(f"[{self.hospital_id}] Getting trainable parameters.")
        return self._get_trainable_parameters()

    def set_parameters(self, parameters):
        """
        Sets the model parameters.
        Handles both full model parameters (initial) and encrypted aggregated updates.
        """
        state_dict_keys = list(self.model.state_dict().keys())

        # Check if parameters are encrypted (serialized uint8 arrays)
        if parameters and len(parameters) > 0 and hasattr(parameters[0], "dtype"):
            if parameters[0].dtype == np.uint8:
                # These are encrypted aggregated gradients
                print(
                    f"[{self.hospital_id}] Received {len(parameters)} encrypted aggregated updates"
                )

                if self.he_manager is None:
                    raise RuntimeError(
                        "Received encrypted updates but HE context not loaded!"
                    )

                # Deserialize encrypted vectors
                encrypted_vectors = self.he_manager.deserialize_vectors(parameters)

                # Decrypt locally
                print(
                    f"[{self.hospital_id}] 🔓 Decrypting aggregated updates locally..."
                )
                decrypted_updates = self.he_manager.decrypt_gradients(encrypted_vectors)

                # Apply decrypted updates to trainable parameters
                trainable_params = [
                    p for p in self.model.parameters() if p.requires_grad
                ]
                trainable_keys = [
                    name for name, p in self.model.named_parameters() if p.requires_grad
                ]

                print(
                    f"[{self.hospital_id}] Applying {len(decrypted_updates)} decrypted updates to model"
                )

                for param, update, key in zip(
                    trainable_params, decrypted_updates, trainable_keys
                ):
                    # Reshape update to match parameter shape
                    # Convert to float32 to match model dtype
                    update_reshaped = torch.tensor(update, dtype=torch.float32).reshape(
                        param.shape
                    )
                    param.data = update_reshaped
                    print(f"   ✓ Updated {key}: {param.shape}")

                print(f"[{self.hospital_id}] ✅ Local decryption and update complete")
                return

        # Otherwise, full model parameters (initial setup or evaluation)
        print(f"[{self.hospital_id}] Received {len(parameters)} full model parameters")

        if len(parameters) != len(state_dict_keys):
            print(f"ERROR: Parameter count mismatch!")
            print(f"  Expected: {len(state_dict_keys)}")
            print(f"  Received: {len(parameters)}")
            raise ValueError(
                f"Parameter count mismatch: got {len(parameters)}, expected {len(state_dict_keys)}"
            )

        params_dict = zip(state_dict_keys, parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self._ensure_he_context(config)
        # IMPORTANT: The client now receives the FULL updated model from the server
        self.set_parameters(parameters)
        self.model.train()

        epoch_loss, correct, total = 0.0, 0, 0

        for batch_idx, (pixel_values, labels) in enumerate(self.trainloader):
            pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 50 == 0 and batch_idx > 0:
                print(
                    f"[{self.hospital_id}] Batch {batch_idx}/{len(self.trainloader)} - Loss: {loss.item():.4f}"
                )

        accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(self.trainloader)

        print(
            f"[{self.hospital_id}] Round finished - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%"
        )

        updated_params = self._get_trainable_parameters()
        payload = self._prepare_parameter_payload(updated_params)

        # Include metadata for server
        metrics = {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "hospital_id": self.hospital_id,
            "encrypted": self.he_manager
            is not None,  # Tell server we're using encryption
        }

        return payload, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self._ensure_he_context(config)
        self.set_parameters(parameters)
        self.model.eval()

        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for pixel_values, labels in self.trainloader:
                pixel_values, labels = (
                    pixel_values.to(self.device),
                    labels.to(self.device),
                )
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                loss += self.criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = loss / len(self.trainloader)

        print(
            f"[{self.hospital_id}] Evaluation - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%"
        )

        return float(avg_loss), total, {"accuracy": float(accuracy)}

    def _ensure_he_context(self, config: dict):
        """Load the HE context delivered by the server if needed."""
        if self.he_manager is not None:
            return
        context_b64 = config.get("he_context_b64") if config else None
        context_id = config.get("he_context_id") if config else None
        if not context_b64:
            print(
                f"[{self.hospital_id}] HE context not provided. Falling back to plaintext updates."
            )
            return
        context_bytes = base64.b64decode(context_b64.encode("ascii"))
        # Load context WITH secret key so client can decrypt aggregated results
        self.he_manager = HEManager.from_serialized(context_bytes, has_secret=True)
        self.he_context_id = context_id
        print(
            f"[{self.hospital_id}] HE context loaded (id={self.he_context_id}). Encrypting updates."
        )

    def _get_trainable_parameters(self):
        return [
            val.cpu().detach().numpy()
            for val in self.model.parameters()
            if val.requires_grad
        ]

    def _prepare_parameter_payload(self, params):
        if self.he_manager is None:
            return params
        encrypted = self.he_manager.encrypt_gradients(params, encrypt_all=False)
        return self.he_manager.serialize_vectors(encrypted)


def start_client(
    hospital_id: str, server_address: str = "localhost:8080", data_root: str = None
):
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Determine data path: use provided data_root or auto-detect
    if data_root is None:
        # Check if running in Docker (look for /app/data)
        import os

        if os.path.exists("/app/data"):
            data_path = f"/app/data/{hospital_id}"
        else:
            # Running locally - use path relative to repo root
            # Script is in backend/fl_client/, so go up two levels to reach repo root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(script_dir))
            data_path = os.path.join(repo_root, "data", hospital_id)
    else:
        # If data_root is provided, use it directly (it might already include hospital_id)
        data_path = data_root

    client = PneumoniaClient(hospital_id, data_path, device=device)

    # Server address: use flower_server for Docker, localhost for local
    if server_address == "localhost:8080":
        import os

        if os.path.exists("/app/data"):
            server_address = "flower_server:8080"

    fl.client.start_client(server_address=server_address, client=client.to_client())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Federated Learning Client for Hospital Data"
    )
    parser.add_argument(
        "--hospital-id",
        type=str,
        default="hospital_1",
        help="Hospital identifier (e.g., hospital_1, hospital_2)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to hospital data directory (optional, will auto-detect if not provided)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:8080",
        help="Server address (default: localhost:8080)",
    )

    args = parser.parse_args()

    start_client(
        hospital_id=args.hospital_id,
        server_address=args.server,
        data_root=args.data_path,
    )
