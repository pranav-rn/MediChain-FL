# medichain-fl/backend/fl_client_integrated.py
"""
Integrated Federated Learning Client with Homomorphic Encryption

This client:
1. Receives global model from server
2. Trains locally on hospital data
3. Extracts partial (trainable) parameters
4. Encrypts gradients using homomorphic encryption
5. Sends encrypted gradients to server
"""

import base64
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from pathlib import Path
import sys
from typing import Optional
import numpy as np

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))
from model import load_model, load_image_processor
from utils.encryption import HEManager


class HuggingFaceImageFolder(Dataset):
    """Wrapper for ImageFolder to apply Hugging Face's image processor"""
    
    def __init__(self, root: str, image_processor):
        self.image_processor = image_processor
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


class BlockchainFLClient(fl.client.NumPyClient):
    """
    Federated Learning Client with Homomorphic Encryption
    
    Features:
    - Trains model locally on hospital data
    - Encrypts partial updates (gradients) using CKKS
    - Sends encrypted gradients to server
    - Receives updated global model
    """
    
    def __init__(
        self,
        hospital_id: str,
        data_path: str,
        model_name: str = "dima806/chest_xray_pneumonia_detection",
        device: str = 'cpu',
        batch_size: int = 8,
        learning_rate: float = 0.001
    ):
        self.hospital_id = hospital_id
        self.device = device
        self.he_manager: Optional[HEManager] = None
        self.he_context_id: Optional[str] = None
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING CLIENT: {hospital_id}")
        print(f"{'='*60}")
        
        # Load model
        print(f"📥 Loading model: {model_name}")
        self.model = load_model(model_name=model_name, freeze_encoder=True).to(device)
        self.image_processor = load_image_processor(model_name=model_name)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"✅ Model loaded")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Frozen: {frozen_params:,}")
        
        # Load dataset
        print(f"\n📂 Loading dataset from: {data_path}")
        self.trainset = HuggingFaceImageFolder(data_path, image_processor=self.image_processor)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"✅ Dataset loaded")
        print(f"   Samples: {len(self.trainset)}")
        print(f"   Batches: {len(self.trainloader)}")
        print(f"   Batch size: {batch_size}")
        
        # Training setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        print(f"\n⚙️  Training configuration")
        print(f"   Optimizer: Adam")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss: CrossEntropyLoss")
        print(f"   Device: {self.device}")
        print(f"{'='*60}\n")

    def get_parameters(self, config):
        """Return trainable parameters"""
        print(f"[{self.hospital_id}] 📤 Sending trainable parameters")
        return self._get_trainable_parameters()
    
    def set_parameters(self, parameters):
        """Set full model parameters from server"""
        print(f"[{self.hospital_id}] 📥 Receiving updated model from server")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train model locally and return encrypted partial updates
        
        Steps:
        1. Load HE context from server
        2. Update model with global parameters
        3. Train locally
        4. Extract partial (trainable) parameters
        5. Encrypt gradients
        6. Return encrypted gradients
        """
        
        round_num = config.get("round_number", 0)
        
        print(f"\n{'='*60}")
        print(f"[{self.hospital_id}] TRAINING ROUND {round_num}")
        print(f"{'='*60}")
        
        # Load HE context if not already loaded
        self._ensure_he_context(config)
        
        # Update model with global parameters
        self.set_parameters(parameters)
        self.model.train()
        
        # Local training
        print(f"🏋️  Training on {len(self.trainset)} samples...")
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
                print(f"   Batch {batch_idx}/{len(self.trainloader)} - Loss: {loss.item():.4f}")
        
        accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(self.trainloader)
        
        print(f"\n✅ Training complete")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.2f}%")
        
        # Get trainable parameters (partial update)
        print(f"\n🔒 Encrypting partial updates...")
        updated_params = self._get_trainable_parameters()
        
        # Encrypt and prepare payload
        payload, encrypted = self._prepare_encrypted_payload(updated_params)
        
        print(f"✅ Encrypted {len(updated_params)} layers")
        print(f"{'='*60}\n")
        
        return payload, len(self.trainset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "hospital_id": self.hospital_id,
            "encrypted": encrypted,
            "round": round_num
        }
    
    def evaluate(self, parameters, config):
        """Evaluate model on local data"""
        
        print(f"\n[{self.hospital_id}] 📊 Evaluating model...")
        
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
        
        print(f"   Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
        
        return float(avg_loss), total, {"accuracy": float(accuracy)}

    def _ensure_he_context(self, config: dict):
        """Load HE public context from server"""
        if self.he_manager is not None:
            return
        
        context_b64 = config.get("he_context_b64")
        context_id = config.get("he_context_id")
        
        if not context_b64:
            print(f"[{self.hospital_id}] ⚠️  No HE context provided - falling back to plaintext")
            return
        
        print(f"[{self.hospital_id}] 🔐 Loading HE context (id={context_id})...")
        context_bytes = base64.b64decode(context_b64.encode("ascii"))
        
        # Create HE manager from serialized public context
        import tenseal as ts
        context = ts.context_from(context_bytes)
        self.he_manager = HEManager()
        self.he_manager.context = context
        self.he_context_id = context_id
        
        print(f"[{self.hospital_id}] ✅ HE context loaded - encryption enabled")

    def _get_trainable_parameters(self):
        """Extract trainable parameters as numpy arrays"""
        return [
            val.cpu().detach().numpy() 
            for val in self.model.parameters() 
            if val.requires_grad
        ]

    def _prepare_encrypted_payload(self, params):
        """Encrypt parameters and prepare for transmission"""
        
        if self.he_manager is None:
            print(f"[{self.hospital_id}] ⚠️  HE not available - sending plaintext")
            return params, False
        
        # Encrypt gradients
        encrypted_vectors = self.he_manager.encrypt_gradients(params, encrypt_all=False)
        
        # Serialize encrypted vectors to bytes
        serialized = []
        for enc_vec in encrypted_vectors:
            enc_bytes = enc_vec.serialize()
            # Convert to numpy array for Flower transmission
            serialized.append(np.frombuffer(enc_bytes, dtype=np.uint8))
        
        return serialized, True


def start_client(
    hospital_id: str,
    data_path: str,
    server_address: str = "localhost:8080",
    device: str = None
):
    """
    Start FL client
    
    Args:
        hospital_id: Unique identifier for this hospital
        data_path: Path to training data
        server_address: FL server address (host:port)
        device: Device to use (cuda/mps/cpu, auto-detected if None)
    """
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"\n{'='*70}")
    print(f"STARTING BLOCKCHAIN FL CLIENT")
    print(f"{'='*70}")
    print(f"Hospital ID: {hospital_id}")
    print(f"Server: {server_address}")
    print(f"Data path: {data_path}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Create client
    client = BlockchainFLClient(
        hospital_id=hospital_id,
        data_path=data_path,
        device=device
    )
    
    # Start Flower client
    print(f"🚀 Connecting to server at {server_address}...\n")
    
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )
    
    print(f"\n{'='*70}")
    print(f"CLIENT {hospital_id} FINISHED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Blockchain FL Client")
    parser.add_argument("--hospital-id", type=str, required=True, help="Hospital identifier")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    start_client(
        hospital_id=args.hospital_id,
        data_path=args.data_path,
        server_address=args.server,
        device=args.device
    )
