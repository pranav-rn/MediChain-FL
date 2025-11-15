# medichain-fl/backend/fl_client/server.py

import base64
import uuid
import hashlib

import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate
import sys
from pathlib import Path

# Add backend to path to import model
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "blockchain" / "python"))
from model import load_model
from utils.encryption import HEManager

# Try to import blockchain client (optional)
try:
    from blockchain_client import log_gradient_update, get_reputation, get_token_balance
    BLOCKCHAIN_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Blockchain integration not available: {e}")
    BLOCKCHAIN_AVAILABLE = False

class PartialUpdateStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters: Parameters, use_homomorphic_encryption: bool = True, **kwargs):
        super().__init__(**kwargs)
        # Store the full set of initial parameters
        self.initial_parameters = initial_parameters
        self.he_enabled = use_homomorphic_encryption
        self.he_manager: Optional[HEManager] = HEManager() if self.he_enabled else None
        self.he_context_serialized: Optional[bytes] = None
        self.he_context_b64: Optional[str] = None
        self.he_context_id: Optional[str] = None
        
        # We need to know which layers are trainable AND the names of ALL layers.
        temp_model = load_model(freeze_encoder=True)
        temp_state = temp_model.state_dict()
        self.trainable_param_keys = [name for name, param in temp_model.named_parameters() if param.requires_grad]
        self.trainable_param_shapes = {name: tuple(temp_state[name].shape) for name in self.trainable_param_keys}
        self.all_param_keys = list(temp_state.keys())

        if self.he_manager:
            self.he_context_serialized = self.he_manager.serialize_context()
            self.he_context_b64 = base64.b64encode(self.he_context_serialized).decode("ascii")
            self.he_context_id = f"ctx-{uuid.uuid4()}"
            print(f"Server Strategy: Homomorphic encryption enabled (context id={self.he_context_id}).")
        else:
            print("Server Strategy: Homomorphic encryption disabled.")

        print(f"Server Strategy: Identified {len(self.all_param_keys)} total layers.")
        print(f"Server Strategy: Identified {len(self.trainable_param_keys)} trainable layers: {self.trainable_param_keys}")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """
        Initialize the global model with the full set of parameters.
        """
        print("Strategy: Initializing global model with full parameters.")
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        fit_configurations = super().configure_fit(server_round, parameters, client_manager)
        if not self.he_manager or not fit_configurations:
            return fit_configurations

        for _, fit_ins in fit_configurations:
            fit_ins.config["he_context_b64"] = self.he_context_b64
            fit_ins.config["he_context_id"] = self.he_context_id or "default"
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results:
            print(f"🔄 ROUND {server_round} - No results from clients. Skipping aggregation.")
            # In case of failure, return the last known good model to continue
            current_full_params = self.initial_parameters if self.initial_parameters else None
            return current_full_params, {}
        
        print(f"\n🔄 ROUND {server_round} - Aggregating {len(results)} PARTIAL client updates.")
        
        # ===== ANOMALY DETECTION: L2 NORM THRESHOLDING =====
        print(f"\n🛡️  ANOMALY DETECTION - Checking gradient norms...")
        client_norms = []
        for client, fit_res in results:
            # Calculate L2 norm of the update
            update_arrays = parameters_to_ndarrays(fit_res.parameters)
            update_vector = np.concatenate([arr.flatten() for arr in update_arrays])
            norm = np.linalg.norm(update_vector)
            client_norms.append((client, norm))
            print(f"   Client {client.cid}: L2 Norm = {norm:.4f}")
        
        # Calculate median-based threshold
        norms_only = [norm for _, norm in client_norms]
        median_norm = np.median(norms_only)
        # Threshold: 2x median (tunable)
        norm_threshold = median_norm * 2.0
        
        print(f"\n   📊 Median Norm: {median_norm:.4f}")
        print(f"   🚨 Threshold (2x median): {norm_threshold:.4f}")
        
        # Filter out anomalous updates
        good_results = []
        flagged_clients = []
        
        for i, (client, fit_res) in enumerate(results):
            client_norm = client_norms[i][1]
            if client_norm > norm_threshold:
                print(f"   🚨 FLAGGED: Client {client.cid} (norm {client_norm:.4f} > {norm_threshold:.4f})")
                flagged_clients.append(client.cid)
            else:
                print(f"   ✅ PASSED: Client {client.cid} (norm {client_norm:.4f} ≤ {norm_threshold:.4f})")
                good_results.append((client, fit_res))
        
        if not good_results:
            print(f"\n⚠️  All clients flagged as anomalous! Skipping aggregation.")
            return self.initial_parameters, {}
        
        print(f"\n✅ Proceeding with {len(good_results)}/{len(results)} validated clients.")
        
        # ===== BLOCKCHAIN LOGGING =====
        if BLOCKCHAIN_AVAILABLE:
            print(f"\n🔗 Logging updates to blockchain...")
            # Log good clients
            for client, fit_res in good_results:
                try:
                    # Hash the client's update
                    update_arrays = parameters_to_ndarrays(fit_res.parameters)
                    update_bytes = b''.join([arr.tobytes() for arr in update_arrays])
                    gradient_hash = hashlib.sha256(update_bytes).hexdigest()
                    
                    # Log to blockchain with hospital_id (client.cid)
                    receipt = log_gradient_update(gradient_hash, flagged=False, hospital_id=client.cid)
                    print(f"   ✅ {client.cid}: Logged to blockchain (Block #{receipt['blockNumber']})")
                except Exception as e:
                    print(f"   ⚠️  Failed to log {client.cid}: {e}")
            
            # Log flagged clients
            for client_id in flagged_clients:
                try:
                    gradient_hash = f"flagged_{client_id}_round_{server_round}"
                    receipt = log_gradient_update(gradient_hash, flagged=True, hospital_id=client_id)
                    print(f"   🚨 {client_id}: Flagged on blockchain (Block #{receipt['blockNumber']})")
                except Exception as e:
                    print(f"   ⚠️  Failed to flag {client_id}: {e}")
        # ===== END BLOCKCHAIN LOGGING =====

        use_encrypted_path = self.he_manager is not None and self._payload_is_encrypted(good_results)
        if use_encrypted_path:
            return self._aggregate_encrypted_fit(server_round, good_results, flagged_clients)
        return self._aggregate_plain_fit(server_round, good_results, flagged_clients)

    def _aggregate_plain_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        flagged_clients: List[str] = None,
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        current_weights = parameters_to_ndarrays(self.initial_parameters)
        state_dict = dict(zip(self.all_param_keys, current_weights))

        partial_updates_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_updates = aggregate(partial_updates_aggregated)

        aggregated_updates_dict = dict(zip(self.trainable_param_keys, aggregated_updates))
        state_dict.update(aggregated_updates_dict)

        updated_full_parameters = ndarrays_to_parameters(list(state_dict.values()))
        self.initial_parameters = updated_full_parameters

        print(f"✅ Round {server_round} - Plain aggregation complete. Full model updated.")
        return updated_full_parameters, self._aggregate_metrics(results)

    def _aggregate_encrypted_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        flagged_clients: List[str] = None,
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        if not self.he_manager:
            raise RuntimeError("HE aggregation requested but HE manager is not initialized")

        print(f"🔐 Round {server_round} - Performing homomorphic aggregation (encrypted domain)")
        
        serialized_updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        sample_counts = [fit_res.num_examples for _, fit_res in results]

        # Deserialize encrypted vectors from clients
        encrypted_updates = [self.he_manager.deserialize_vectors(update) for update in serialized_updates]
        
        # CRITICAL: Aggregate in encrypted domain - server never sees plaintext!
        aggregated_encrypted = self.he_manager.aggregate_encrypted_weighted(encrypted_updates, sample_counts)
        
        # Serialize the ENCRYPTED aggregated result (DO NOT DECRYPT!)
        serialized_encrypted = self.he_manager.serialize_vectors(aggregated_encrypted)
        
        # Convert to Parameters to send back to clients
        # Clients will receive ENCRYPTED aggregates and decrypt locally
        encrypted_parameters = ndarrays_to_parameters(serialized_encrypted)
        
        # Store encrypted aggregates (server never decrypts!)
        self.initial_parameters = encrypted_parameters

        print(f"🔐 Round {server_round} - Homomorphic aggregation complete.")
        print(f"   ✅ Server NEVER decrypted the weights - privacy preserved!")
        print(f"   ✅ Encrypted aggregated weights sent back to clients")
        
        return encrypted_parameters, self._aggregate_metrics(results)

    @staticmethod
    def _aggregate_metrics(
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]
    ) -> Dict[str, fl.common.Scalar]:
        aggregated_metrics: Dict[str, fl.common.Scalar] = {}
        if not results:
            return aggregated_metrics
        accuracies = [res.metrics.get("accuracy", 0) for _, res in results]
        if accuracies:
            aggregated_metrics["accuracy_avg"] = sum(accuracies) / len(accuracies)
        return aggregated_metrics

    def _payload_is_encrypted(
        self,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> bool:
        sample_arrays = parameters_to_ndarrays(results[0][1].parameters)
        return bool(sample_arrays) and sample_arrays[0].dtype == np.uint8

def start_server(num_rounds: int = 5, use_he: bool = True):
    """Starts the Flower server with our custom partial update strategy."""
    
    # The server must load the model to get the initial state
    print("Server: Loading initial model to create strategy...")
    initial_model = load_model(freeze_encoder=True)
    initial_params_list = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_params_list)

    # Instantiate our custom strategy
    strategy = PartialUpdateStrategy(
        initial_parameters=initial_parameters,
        use_homomorphic_encryption=use_he,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy, 
    )

if __name__ == "__main__":
    import sys
    use_he = "--no-he" not in sys.argv
    start_server(num_rounds=5, use_he=use_he)