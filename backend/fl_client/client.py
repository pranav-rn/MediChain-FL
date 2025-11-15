# medichain-fl/backend/fl_client/client.py
import base64
import flwr as fl
import torch
import torch.nn as nn
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
        pixel_values = processed_inputs['pixel_values'].squeeze(0)
        return pixel_values, label


class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, data_path: str, model_name: str = "dima806/chest_xray_pneumonia_detection", device: str = 'cpu'):
        self.hospital_id = hospital_id
        self.device = device
        self.he_manager: Optional[HEManager] = None
        self.he_context_id: str | None = None
        
        # Load Hugging Face model
        self.model = load_model(model_name=model_name, freeze_encoder=True).to(device)
        
        # Load processor
        self.image_processor = load_image_processor(model_name=model_name)
        
        # Dataset
        self.trainset = HuggingFaceImageFolder(data_path, image_processor=self.image_processor)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=8, # Kept the reduced batch size
            shuffle=True,
            num_workers=0, # IMPORTANT CHANGE: Set num_workers to 0 for stability on macOS/Docker
            pin_memory=False # IMPORTANT CHANGE: Set pin_memory to False when not using a GPU
        )
        
        # Loss + optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.001
        )

        print(f"[{self.hospital_id}] Initialized with {len(self.trainset)} samples. Device: {self.device}")
        print(f"[{self.hospital_id}] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def get_parameters(self, config):
        """
        Returns only the model parameters that are trainable (unfrozen).
        """
        print(f"[{self.hospital_id}] Getting trainable parameters.")
        return self._get_trainable_parameters()
    
    def set_parameters(self, parameters):
        """
        Sets the model parameters.
        NOTE: This client now receives the FULL model from the server,
        so this function can handle loading the full state dict correctly.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
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
                print(f"[{self.hospital_id}] Batch {batch_idx}/{len(self.trainloader)} - Loss: {loss.item():.4f}")
        
        accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(self.trainloader)
        
        print(f"[{self.hospital_id}] Round finished - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        updated_params = self._get_trainable_parameters()
        payload = self._prepare_parameter_payload(updated_params)
        
        return payload, len(self.trainloader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy)
        }
    
    def evaluate(self, parameters, config):
        self._ensure_he_context(config)
        self.set_parameters(parameters)
        self.model.eval()
        
        loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for pixel_values, labels in self.trainloader:
                pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                loss += self.criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = loss / len(self.trainloader)
        
        print(f"[{self.hospital_id}] Evaluation - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        return float(avg_loss), total, {"accuracy": float(accuracy)}

    def _ensure_he_context(self, config: dict):
        """Load the HE public context delivered by the server if needed."""
        if self.he_manager is not None:
            return
        context_b64 = config.get("he_context_b64") if config else None
        context_id = config.get("he_context_id") if config else None
        if not context_b64:
            print(f"[{self.hospital_id}] HE context not provided. Falling back to plaintext updates.")
            return
        context_bytes = base64.b64decode(context_b64.encode("ascii"))
        self.he_manager = HEManager.from_serialized(context_bytes, has_secret=False)
        self.he_context_id = context_id
        print(f"[{self.hospital_id}] HE context loaded (id={self.he_context_id}). Encrypting updates.")

    def _get_trainable_parameters(self):
        return [val.cpu().detach().numpy() for val in self.model.parameters() if val.requires_grad]

    def _prepare_parameter_payload(self, params):
        if self.he_manager is None:
            return params
        encrypted = self.he_manager.encrypt_gradients(params, encrypt_all=False)
        return self.he_manager.serialize_vectors(encrypted)

def start_client(hospital_id: str, server_address: str = "localhost:8080", data_root: str = None):
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
        data_path = f"{data_root}/{hospital_id}"

    client = PneumoniaClient(
        hospital_id,
        data_path,
        device=device
    )

    # Server address: use flower_server for Docker, localhost for local
    if server_address == "localhost:8080":
        import os
        if os.path.exists("/app/data"):
            server_address = "flower_server:8080"
    
    fl.client.start_client(server_address=server_address, client=client.to_client())

if __name__ == "__main__":
    import sys
    hospital_id = sys.argv[1] if len(sys.argv) > 1 else "hospital_1"
    start_client(hospital_id)