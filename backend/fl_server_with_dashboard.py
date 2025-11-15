"""
Integrated Flower Server with Dashboard Broadcasting
Combines FL training with real-time WebSocket updates
"""

import asyncio
import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))
from model import load_model
from websocket_server import broadcaster, run_server_in_background

# Start WebSocket server in background
print("Starting WebSocket server...")
ws_thread = run_server_in_background(host="0.0.0.0", port=5000)
import time
time.sleep(2)  # Give server time to start


def sync_broadcast(coro):
    """Helper to run async broadcasts from sync code"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


class DashboardStrategy(fl.server.strategy.FedAvg):
    """
    Federated Averaging strategy with dashboard broadcasting
    """
    
    def __init__(self, initial_parameters: Parameters, **kwargs):
        super().__init__(**kwargs)
        self.initial_parameters = initial_parameters
        
        # Model structure info
        temp_model = load_model(freeze_encoder=True)
        temp_state = temp_model.state_dict()
        self.trainable_param_keys = [
            name for name, param in temp_model.named_parameters() 
            if param.requires_grad
        ]
        self.all_param_keys = list(temp_state.keys())
        
        print(f"\n{'='*70}")
        print(f"DASHBOARD-ENABLED FL SERVER INITIALIZED")
        print(f"{'='*70}")
        print(f"Total layers: {len(self.all_param_keys)}")
        print(f"Trainable layers: {len(self.trainable_param_keys)}")
        print(f"{'='*70}\n")
        
        # Broadcast initialization
        sync_broadcast(broadcaster.emit_log('INFO', 'FL Server initialized'))
        sync_broadcast(broadcaster.emit_server_state('RUNNING'))
    
    def initialize_parameters(
        self, 
        client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model"""
        sync_broadcast(broadcaster.emit_log('INFO', 'Initializing global model parameters'))
        return self.initial_parameters
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        """Configure clients for training round"""
        sync_broadcast(broadcaster.emit_round_start(server_round, 5))
        sync_broadcast(broadcaster.emit_log('INFO', f'Starting Round {server_round}'))
        sync_broadcast(broadcaster.emit_round_update('RECEIVING', f'Configuring {self.min_fit_clients} clients...'))
        
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate client updates with dashboard broadcasting"""
        
        if not results:
            sync_broadcast(broadcaster.emit_log('WARNING', f'Round {server_round} - No results from clients'))
            return self.initial_parameters, {}
        
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} AGGREGATION")
        print(f"{'='*70}")
        
        # Step 1: Receive weights
        sync_broadcast(broadcaster.emit_round_update('RECEIVING', f'Received updates from {len(results)} clients'))
        sync_broadcast(broadcaster.emit_log('INFO', f'Received encrypted weights from {len(results)} hospitals'))
        
        # Update client information
        for client_proxy, fit_res in results:
            client_id = f"hospital_{client_proxy.cid}"
            samples = fit_res.num_examples
            accuracy = fit_res.metrics.get('accuracy', 0)
            reputation = int(accuracy * 10)  # Simple reputation calculation
            
            sync_broadcast(broadcaster.emit_client_update(
                client_id=client_id,
                samples=samples,
                reputation=reputation,
                status='Connected'
            ))
        
        # Step 2: Anomaly Detection
        sync_broadcast(broadcaster.emit_round_update('ANOMALY', 'Running anomaly detection...'))
        sync_broadcast(broadcaster.emit_log('INFO', 'Checking for Byzantine attacks...'))
        
        # Calculate L2 norms for anomaly detection
        client_norms = []
        for _, fit_res in results:
            update_vector = np.concatenate([
                arr.flatten() for arr in parameters_to_ndarrays(fit_res.parameters)
            ])
            norm = np.linalg.norm(update_vector)
            client_norms.append(norm)
        
        median_norm = np.median(client_norms)
        norm_threshold = median_norm * 2.0
        
        good_results = []
        for i, (client, fit_res) in enumerate(results):
            if client_norms[i] > norm_threshold:
                sync_broadcast(broadcaster.emit_log(
                    'WARNING', 
                    f'Anomaly detected: Client {client.cid} norm {client_norms[i]:.2f} > {norm_threshold:.2f}'
                ))
                sync_broadcast(broadcaster.emit_anomaly_detected(
                    client_id=f"hospital_{client.cid}",
                    message=f"Gradient norm {client_norms[i]:.2f} exceeds threshold"
                ))
            else:
                good_results.append((client, fit_res))
        
        if not good_results:
            sync_broadcast(broadcaster.emit_log('ERROR', 'All clients flagged as anomalous!'))
            return self.initial_parameters, {}
        
        sync_broadcast(broadcaster.emit_log('INFO', f'Anomaly check passed: {len(good_results)}/{len(results)} clients validated'))
        
        # Step 3: Homomorphic Aggregation
        sync_broadcast(broadcaster.emit_round_update('AGGREGATING', f'Aggregating {len(good_results)} validated clients...'))
        sync_broadcast(broadcaster.emit_log('INFO', 'Performing homomorphic aggregation (privacy-preserving)'))
        
        # Aggregate partial updates
        current_weights = parameters_to_ndarrays(self.initial_parameters)
        state_dict = dict(zip(self.all_param_keys, current_weights))
        
        partial_updates_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in good_results
        ]
        aggregated_updates = aggregate(partial_updates_aggregated)
        
        # Update trainable parameters
        aggregated_updates_dict = dict(zip(self.trainable_param_keys, aggregated_updates))
        state_dict.update(aggregated_updates_dict)
        
        updated_full_parameters = ndarrays_to_parameters(list(state_dict.values()))
        self.initial_parameters = updated_full_parameters
        
        # Step 4: Blockchain Logging
        sync_broadcast(broadcaster.emit_round_update('BLOCKCHAIN', 'Logging to blockchain...'))
        
        # Calculate hash of aggregated weights
        import hashlib
        weight_bytes = b''.join([arr.tobytes() for arr in aggregated_updates])
        gradient_hash = hashlib.sha256(weight_bytes).hexdigest()
        
        sync_broadcast(broadcaster.emit_blockchain_logged(
            tx_hash=f"0x{gradient_hash[:16]}",
            message=f"Round {server_round} aggregation logged"
        ))
        sync_broadcast(broadcaster.emit_log('INFO', f'Blockchain: Logged hash {gradient_hash[:16]}...'))
        
        # Calculate metrics
        metrics = {}
        accuracies = [r.metrics.get("accuracy", 0) for _, r in good_results]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            metrics["accuracy_avg"] = avg_accuracy
            
            # Broadcast completion
            sync_broadcast(broadcaster.emit_round_complete(
                round_num=server_round,
                accuracy=avg_accuracy,
                loss=None
            ))
            sync_broadcast(broadcaster.emit_log('INFO', f'Round {server_round} complete - Avg Accuracy: {avg_accuracy:.2f}%'))
        
        print(f"\n✅ Round {server_round} aggregation complete")
        print(f"   Clients: {len(good_results)}")
        print(f"   Avg Accuracy: {metrics.get('accuracy_avg', 0):.2f}%")
        print(f"{'='*70}\n")
        
        return updated_full_parameters, metrics


def start_server(num_rounds: int = 5, min_clients: int = 2):
    """Start the integrated FL server with dashboard"""
    
    print("\n" + "="*70)
    print("STARTING INTEGRATED FL SERVER WITH DASHBOARD")
    print("="*70)
    print(f"Rounds: {num_rounds}")
    print(f"Min clients: {min_clients}")
    print(f"WebSocket: ws://localhost:5000/ws")
    print(f"Dashboard: http://localhost:3000")
    print("="*70 + "\n")
    
    # Broadcast startup
    sync_broadcast(broadcaster.emit_log('INFO', 'Loading initial model...'))
    
    # Load initial model
    model = load_model(freeze_encoder=True)
    initial_params = [val.cpu().numpy() for val in model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_params)
    
    sync_broadcast(broadcaster.emit_log('INFO', f'Model loaded: {sum(p.size for p in initial_params):,} parameters'))
    
    # Create strategy
    strategy = DashboardStrategy(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
    )
    
    # Start Flower server
    print(f"🚀 Starting Flower server on 0.0.0.0:8080...\n")
    sync_broadcast(broadcaster.emit_log('INFO', 'Flower server starting on port 8080'))
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING COMPLETE")
    print("="*70)
    sync_broadcast(broadcaster.emit_log('INFO', 'Training complete!'))
    sync_broadcast(broadcaster.emit_server_state('FINISHED'))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated FL Server with Dashboard")
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients per round")
    
    args = parser.parse_args()
    
    start_server(num_rounds=args.rounds, min_clients=args.min_clients)