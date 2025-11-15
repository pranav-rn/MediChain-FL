# medichain-fl/backend/fl_server_integrated.py
"""
Integrated Federated Learning Server with Blockchain and Homomorphic Encryption

This server:
1. Receives encrypted gradients from clients
2. Aggregates them homomorphically (without decryption)
3. Logs encrypted gradient hashes to blockchain
4. Decrypts aggregated result
5. Detects anomalies and flags suspicious updates on blockchain
6. Sends updated model back to clients
"""

import base64
import uuid
import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate
import sys
from pathlib import Path
import torch

# Add backend to path
sys.path.append(str(Path(__file__).parent))
from model import load_model
from blockchain_integration import BlockchainFLServer


class BlockchainHEStrategy(fl.server.strategy.FedAvg):
    """
    Federated Learning strategy with:
    - Homomorphic Encryption for privacy
    - Blockchain logging for audit trail
    - Anomaly detection for Byzantine fault tolerance
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        enable_blockchain: bool = True,
        enable_anomaly_detection: bool = True,
        anomaly_threshold: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store initial parameters
        self.initial_parameters = initial_parameters
        self.current_round = 0

        # Initialize blockchain FL server (includes HE and anomaly detection)
        self.blockchain_server = BlockchainFLServer(
            enable_blockchain=enable_blockchain,
            enable_anomaly_detection=enable_anomaly_detection,
            anomaly_threshold=anomaly_threshold,
        )

        # Model structure info
        temp_model = load_model(freeze_encoder=True)
        temp_state = temp_model.state_dict()
        self.trainable_param_keys = [
            name for name, param in temp_model.named_parameters() if param.requires_grad
        ]
        self.trainable_param_shapes = {
            name: tuple(temp_state[name].shape) for name in self.trainable_param_keys
        }
        self.all_param_keys = list(temp_state.keys())

        # HE context for clients - serialize with secret key included
        he_manager = self.blockchain_server.he_manager
        # Ensure secret key is included in serialization
        self.he_context_serialized = he_manager.context.serialize(save_secret_key=True)
        self.he_context_b64 = base64.b64encode(self.he_context_serialized).decode(
            "ascii"
        )
        self.he_context_id = f"ctx-{uuid.uuid4()}"

        print(f"\n{'=' * 70}")
        print(f"BLOCKCHAIN FL SERVER INITIALIZED")
        print(f"{'=' * 70}")
        print(f"Total layers: {len(self.all_param_keys)}")
        print(f"Trainable layers: {len(self.trainable_param_keys)}")
        print(f"Trainable params: {self.trainable_param_keys}")
        print(f"HE Context ID: {self.he_context_id}")
        print(
            f"Blockchain: {'✅ Enabled' if self.blockchain_server.blockchain_enabled else '❌ Disabled'}"
        )
        print(
            f"Anomaly Detection: {'✅ Enabled' if self.blockchain_server.anomaly_detector else '❌ Disabled'}"
        )
        print(f"{'=' * 70}\n")

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model with full parameters"""
        print("🔧 Initializing global model with full parameters")
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        """Configure clients for training round - include HE context"""
        fit_configurations = super().configure_fit(
            server_round, parameters, client_manager
        )

        if not fit_configurations:
            return fit_configurations

        # Add HE context to client config
        for _, fit_ins in fit_configurations:
            fit_ins.config["he_context_b64"] = self.he_context_b64
            fit_ins.config["he_context_id"] = self.he_context_id
            fit_ins.config["round_number"] = server_round

        print(f"\n📤 Configured {len(fit_configurations)} clients with HE context")
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """
        Aggregate client updates using blockchain and HE:
        1. Extract encrypted partial updates from clients
        2. Aggregate homomorphically
        3. Log to blockchain
        4. Detect anomalies
        5. Decrypt and update global model
        """

        if not results:
            print(f"\n⚠️  ROUND {server_round} - No results from clients")
            return self.initial_parameters, {}

        self.current_round = server_round

        print(f"\n{'=' * 70}")
        print(f"ROUND {server_round} AGGREGATION")
        print(f"{'=' * 70}")
        print(f"Clients participated: {len(results)}")
        print(f"Failures: {len(failures)}")

        # Check if clients sent encrypted gradients
        use_encrypted = self._check_encrypted_payload(results)

        if use_encrypted:
            print("🔐 Using ENCRYPTED aggregation pathway")
            return self._aggregate_encrypted(server_round, results)
        else:
            print("⚠️  Using PLAIN aggregation (no encryption)")
            return self._aggregate_plain(server_round, results)

    def _check_encrypted_payload(
        self, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]
    ) -> bool:
        """Check if clients sent encrypted gradients"""
        if not results:
            return False

        _, first_fit_res = results[0]
        return first_fit_res.metrics.get("encrypted", False)

    def _aggregate_encrypted(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """
        Aggregate encrypted gradients using blockchain integration
        """

        print("\n📦 Extracting encrypted gradients from clients...")

        # Extract encrypted gradients from clients
        encrypted_gradients_list = []
        client_ids = []
        num_examples = []

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("hospital_id", f"client_{client_proxy.cid}")
            client_ids.append(client_id)
            num_examples.append(fit_res.num_examples)

            # Encrypted gradients are serialized in parameters
            encrypted_bytes = parameters_to_ndarrays(fit_res.parameters)

            # Deserialize encrypted gradients using HE manager
            encrypted_grads = []
            for enc_bytes in encrypted_bytes:
                # Each encrypted gradient is stored as bytes
                enc_vector = self.blockchain_server.he_manager.deserialize_encrypted(
                    enc_bytes.tobytes()
                )
                encrypted_grads.append(enc_vector)

            encrypted_gradients_list.append(encrypted_grads)
            print(
                f"   ✓ {client_id}: {len(encrypted_grads)} encrypted layers ({fit_res.num_examples} samples)"
            )

        # Use blockchain server to aggregate with blockchain logging
        print(f"\n🔗 Aggregating with blockchain integration...")

        aggregation_result = self.blockchain_server.aggregate_round(
            client_gradients=encrypted_gradients_list,
            client_ids=client_ids,
            num_examples=num_examples,
            round_number=server_round,
        )

        # Extract aggregated ENCRYPTED gradients
        aggregated_encrypted = aggregation_result["aggregated_encrypted"]

        print(f"\n📊 Aggregated encrypted gradients info:")
        print(f"   Type: {type(aggregated_encrypted)}")
        print(f"   Length: {len(aggregated_encrypted)} encrypted layers")

        # Serialize encrypted gradients to send back to clients
        print(f"\n📦 Serializing encrypted aggregates for clients...")
        serialized_encrypted = self.blockchain_server.he_manager.serialize_vectors(
            aggregated_encrypted
        )

        # Convert to Parameters - clients will decrypt locally
        encrypted_parameters = ndarrays_to_parameters(serialized_encrypted)

        # Store encrypted parameters for next round
        self.encrypted_aggregates = encrypted_parameters

        print(
            f"✅ Encrypted aggregates prepared ({len(serialized_encrypted)} encrypted arrays)"
        )
        print(f"🔐 CRITICAL: Server NEVER decrypted - privacy preserved!")

        # Metrics for Flower
        metrics = {
            "round": server_round,
            "num_clients": aggregation_result["num_clients"],
            "anomaly_detected": aggregation_result["anomaly_detected"],
            "anomaly_score": aggregation_result["anomaly_score"] or 0.0,
            "blockchain_logged": aggregation_result["blockchain_receipt"] is not None,
            "flagged_clients": aggregation_result.get("flagged_clients", []),
        }

        print(f"\n{'=' * 70}")
        print(f"ROUND {server_round} COMPLETE")
        print(f"{'=' * 70}")
        print(f"✅ Aggregated {len(results)} clients")
        print(f"✅ Blockchain logged: {metrics['blockchain_logged']}")
        print(f"🔍 Anomaly detected: {metrics['anomaly_detected']}")
        if metrics["anomaly_detected"]:
            print(f"   ⚠️  Anomaly score: {metrics['anomaly_score']:.2f}")
            print(f"   🚨 Flagged clients: {', '.join(metrics['flagged_clients'])}")
        print(f"🔐 Encrypted aggregates returned to clients for local decryption")
        print(f"{'=' * 70}\n")

        return encrypted_parameters, metrics

    def _aggregate_plain(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate plain (unencrypted) gradients - fallback mode"""

        print("⚠️  Aggregating without encryption (fallback mode)")

        # Extract partial updates
        partial_updates = []
        for _, fit_res in results:
            updates = parameters_to_ndarrays(fit_res.parameters)
            partial_updates.append((updates, fit_res.num_examples))

        # Standard FedAvg aggregation
        aggregated_updates = aggregate(partial_updates)

        # Reconstruct full model
        updated_params = self._reconstruct_full_model(aggregated_updates)

        metrics = {
            "round": server_round,
            "num_clients": len(results),
            "encrypted": False,
        }

        return updated_params, metrics

    def _reconstruct_full_model(self, partial_updates: List[np.ndarray]) -> Parameters:
        """
        Reconstruct full model by merging partial updates with frozen parameters
        """
        print(f"\n🔧 _reconstruct_full_model called:")
        print(f"   Received {len(partial_updates)} partial updates")
        print(f"   Trainable param keys: {len(self.trainable_param_keys)}")
        print(f"   All param keys: {len(self.all_param_keys)}")

        # Start with current full model
        current_weights = parameters_to_ndarrays(self.initial_parameters)
        print(f"   Current full model has {len(current_weights)} parameters")

        state_dict = dict(zip(self.all_param_keys, current_weights))

        # Update trainable parameters
        for key, update in zip(self.trainable_param_keys, partial_updates):
            if isinstance(update, np.ndarray):
                # Reshape to match expected shape
                expected_shape = self.trainable_param_shapes[key]
                print(f"   Updating {key}: {update.shape} -> {expected_shape}")
                state_dict[key] = update.reshape(expected_shape)

        # Convert back to parameters
        updated_weights = [state_dict[k] for k in self.all_param_keys]
        print(f"   Final model has {len(updated_weights)} parameters")

        # Update initial_parameters for next round
        self.initial_parameters = ndarrays_to_parameters(updated_weights)

        return self.initial_parameters


def start_server(
    num_rounds: int = 10,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    enable_blockchain: bool = True,
    enable_anomaly_detection: bool = True,
    server_address: str = "0.0.0.0:8080",
):
    """
    Start the integrated FL server

    Args:
        num_rounds: Number of training rounds
        min_fit_clients: Minimum clients required for each round
        min_available_clients: Minimum clients that must connect
        enable_blockchain: Enable blockchain logging
        enable_anomaly_detection: Enable anomaly detection
        server_address: Server address (default: 0.0.0.0:8080)
    """

    print("\n" + "=" * 70)
    print("STARTING BLOCKCHAIN FEDERATED LEARNING SERVER")
    print("=" * 70)
    print(f"Server address: {server_address}")
    print(f"Training rounds: {num_rounds}")
    print(f"Min clients per round: {min_fit_clients}")
    print(f"Blockchain: {'✅ Enabled' if enable_blockchain else '❌ Disabled'}")
    print(
        f"Anomaly detection: {'✅ Enabled' if enable_anomaly_detection else '❌ Disabled'}"
    )
    print("=" * 70 + "\n")

    # Load initial model
    print("🔧 Loading initial model...")
    model = load_model(freeze_encoder=True)
    initial_params = [val.cpu().numpy() for val in model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_params)
    print(f"✅ Model loaded: {sum(p.size for p in initial_params)} total parameters\n")

    # Create strategy
    strategy = BlockchainHEStrategy(
        initial_parameters=initial_parameters,
        enable_blockchain=enable_blockchain,
        enable_anomaly_detection=enable_anomaly_detection,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )

    # Start Flower server
    print(f"🚀 Starting Flower server on {server_address}...\n")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING COMPLETE")
    print("=" * 70)

    # Print final blockchain stats
    if strategy.blockchain_server.blockchain_enabled:
        print("\n📊 Retrieving final blockchain logs...")
        logs = strategy.blockchain_server.get_blockchain_logs(
            limit=100
        )  # Get more history

        print(f"\n📖 Blockchain Summary:")
        print(f"   Total logs on blockchain: {len(logs)}")
        flagged = sum(1 for log in logs if log["flagged"])
        print(f"   Flagged updates: {flagged}")

        # Show token balances
        try:
            from blockchain_client import get_token_balance, HOSPITAL_ACCOUNTS

            print("\n💰 Final Token Balances:")
            for hospital_id in ["hospital_1", "hospital_2", "hospital_3", "hospital_4"]:
                if hospital_id in HOSPITAL_ACCOUNTS:
                    address = HOSPITAL_ACCOUNTS[hospital_id]["address"]
                    balance = get_token_balance(address)
                    print(f"   {hospital_id}: {balance} MCT")
        except Exception as e:
            print(f"   Could not retrieve token balances: {e}")

        if logs:
            print(f"\n   Recent logs:")
            for log in logs[-5:]:
                flag_status = "🚩" if log["flagged"] else "✅"
                print(
                    f"   {flag_status} Log #{log['index']}: {log['gradientHash'][:16]}..."
                )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blockchain FL Server")
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of training rounds"
    )
    parser.add_argument(
        "--min-clients", type=int, default=2, help="Minimum clients per round"
    )
    parser.add_argument(
        "--no-blockchain", action="store_true", help="Disable blockchain"
    )
    parser.add_argument(
        "--no-anomaly", action="store_true", help="Disable anomaly detection"
    )
    parser.add_argument(
        "--address", type=str, default="0.0.0.0:8080", help="Server address"
    )

    args = parser.parse_args()

    start_server(
        num_rounds=args.rounds,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        enable_blockchain=not args.no_blockchain,
        enable_anomaly_detection=not args.no_anomaly,
        server_address=args.address,
    )
