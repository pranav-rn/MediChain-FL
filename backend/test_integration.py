"""
Test Suite for Blockchain + Homomorphic Encryption Integration

This script tests the complete workflow:
1. Simulate multiple clients with gradients
2. Encrypt gradients using homomorphic encryption
3. Aggregate encrypted gradients on server
4. Log to blockchain
5. Detect anomalies
6. Retrieve blockchain logs

Requirements:
- Hardhat node running (npx hardhat node)
- Contract deployed (npx hardhat run scripts/deploy_proxy.js --network localhost)
- Python packages: tenseal, web3, numpy, torch
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from blockchain_integration import BlockchainFLServer


def create_simple_model():
    """Create a simple neural network for testing"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )


def simulate_client_gradients(num_clients: int, model: nn.Module, add_anomaly: bool = False):
    """
    Simulate gradient updates from multiple clients
    
    Args:
        num_clients: Number of clients to simulate
        model: PyTorch model to extract gradient shapes
        add_anomaly: If True, make one client's gradients anomalous
    
    Returns:
        List of gradient lists (one per client)
    """
    print(f"\n📊 Simulating {num_clients} clients...")
    
    # Get model parameter shapes
    param_shapes = [p.shape for p in model.parameters()]
    
    client_gradients = []
    
    for i in range(num_clients):
        gradients = []
        
        for shape in param_shapes:
            if add_anomaly and i == 0:
                # First client has anomalous gradients (very large values)
                grad = np.random.randn(*shape) * 100  # 100x larger
                print(f"   Client {i+1}: ANOMALOUS gradients (scale: 100x)")
            else:
                # Normal gradients
                grad = np.random.randn(*shape) * 0.01  # Small gradients typical in training
            
            gradients.append(grad)
        
        client_gradients.append(gradients)
    
    print(f"✅ Generated gradients for {num_clients} clients")
    print(f"   Layers per client: {len(gradients)}")
    print(f"   Total parameters: {sum(np.prod(s) for s in param_shapes)}")
    
    return client_gradients


def test_basic_aggregation():
    """Test 1: Basic homomorphic aggregation without blockchain"""
    print("\n" + "=" * 70)
    print("TEST 1: BASIC HOMOMORPHIC AGGREGATION")
    print("=" * 70)
    
    # Initialize server without blockchain
    server = BlockchainFLServer(
        enable_blockchain=False,
        enable_anomaly_detection=False
    )
    
    # Create model and simulate clients
    model = create_simple_model()
    client_gradients = simulate_client_gradients(num_clients=3, model=model)
    
    # Perform aggregation
    result = server.aggregate_round(
        client_gradients=client_gradients,
        client_ids=['hospital_1', 'hospital_2', 'hospital_3'],
        round_number=1
    )
    
    # Verify results
    print("\n📊 RESULTS:")
    print(f"   Aggregated {result['num_clients']} clients")
    print(f"   Gradient hash: {result['gradient_hash'][:32]}...")
    print(f"   HE encryptions: {result['he_stats']['encryptions_performed']}")
    print(f"   HE aggregations: {result['he_stats']['aggregations_performed']}")
    print(f"   HE decryptions: {result['he_stats']['decryptions_performed']}")
    
    return result


def test_blockchain_integration():
    """Test 2: Full blockchain integration"""
    print("\n" + "=" * 70)
    print("TEST 2: BLOCKCHAIN INTEGRATION")
    print("=" * 70)
    print("NOTE: Requires Hardhat node and deployed contract")
    print("=" * 70)
    
    # Initialize server with blockchain
    server = BlockchainFLServer(
        enable_blockchain=True,
        enable_anomaly_detection=False
    )
    
    if not server.blockchain_enabled:
        print("\n⚠️  SKIPPED: Blockchain not available")
        print("   Start Hardhat: npx hardhat node")
        print("   Deploy contract: npx hardhat run scripts/deploy_proxy.js --network localhost")
        return None
    
    # Create model and simulate clients
    model = create_simple_model()
    client_gradients = simulate_client_gradients(num_clients=5, model=model)
    
    # Perform aggregation with blockchain logging
    result = server.aggregate_round(
        client_gradients=client_gradients,
        client_ids=[f'hospital_{i}' for i in range(1, 6)],
        round_number=1
    )
    
    # Verify blockchain receipt
    if result['blockchain_receipt']:
        print("\n📊 BLOCKCHAIN RESULTS:")
        print(f"   Transaction: {result['blockchain_receipt']['transactionHash'].hex()}")
        print(f"   Block: {result['blockchain_receipt']['blockNumber']}")
        print(f"   Gas used: {result['blockchain_receipt']['gasUsed']}")
    
    # Retrieve logs from blockchain
    print("\n📖 Retrieving blockchain logs...")
    logs = server.get_blockchain_logs(limit=5)
    
    for log in logs:
        flag_status = "🚩 FLAGGED" if log['flagged'] else "✅ OK"
        print(f"\n   Log #{log['index']} {flag_status}")
        print(f"      Hospital: {log['hospital']}")
        print(f"      Hash: {log['gradientHash'][:32]}...")
        print(f"      Timestamp: {log['timestamp']}")
    
    return result


def test_anomaly_detection():
    """Test 3: Anomaly detection and blockchain flagging"""
    print("\n" + "=" * 70)
    print("TEST 3: ANOMALY DETECTION")
    print("=" * 70)
    
    # Initialize server with anomaly detection
    server = BlockchainFLServer(
        enable_blockchain=True,
        enable_anomaly_detection=True,
        anomaly_threshold=2.0  # Lower threshold for testing
    )
    
    # Create model
    model = create_simple_model()
    
    # Round 1: Normal gradients (to build baseline)
    print("\n--- Round 1: Building Baseline ---")
    normal_gradients = simulate_client_gradients(num_clients=3, model=model, add_anomaly=False)
    result1 = server.aggregate_round(
        client_gradients=normal_gradients,
        client_ids=['hospital_A', 'hospital_B', 'hospital_C'],
        round_number=1
    )
    
    # Round 2: Another normal round
    print("\n--- Round 2: Normal Update ---")
    normal_gradients2 = simulate_client_gradients(num_clients=3, model=model, add_anomaly=False)
    result2 = server.aggregate_round(
        client_gradients=normal_gradients2,
        client_ids=['hospital_A', 'hospital_B', 'hospital_C'],
        round_number=2
    )
    
    # Round 3: Anomalous gradients
    print("\n--- Round 3: Malicious Update (Anomaly) ---")
    anomalous_gradients = simulate_client_gradients(num_clients=3, model=model, add_anomaly=True)
    result3 = server.aggregate_round(
        client_gradients=anomalous_gradients,
        client_ids=['hospital_A_MALICIOUS', 'hospital_B', 'hospital_C'],
        round_number=3
    )
    
    # Summary
    print("\n📊 ANOMALY DETECTION SUMMARY:")
    print(f"   Round 1 - Anomaly: {result1['anomaly_detected']}, Score: {result1['anomaly_score']:.2f}")
    print(f"   Round 2 - Anomaly: {result2['anomaly_detected']}, Score: {result2['anomaly_score']:.2f}")
    print(f"   Round 3 - Anomaly: {result3['anomaly_detected']}, Score: {result3['anomaly_score']:.2f}")
    
    if server.blockchain_enabled:
        print("\n📖 Checking blockchain for flagged updates...")
        logs = server.get_blockchain_logs(limit=10)
        flagged_count = sum(1 for log in logs if log['flagged'])
        print(f"   Total logs: {len(logs)}")
        print(f"   Flagged logs: {flagged_count}")
    
    return result3


def test_multiple_rounds():
    """Test 4: Multiple training rounds"""
    print("\n" + "=" * 70)
    print("TEST 4: MULTIPLE TRAINING ROUNDS")
    print("=" * 70)
    
    server = BlockchainFLServer(
        enable_blockchain=True,
        enable_anomaly_detection=True
    )
    
    model = create_simple_model()
    
    results = []
    for round_num in range(1, 6):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/5")
        print(f"{'='*60}")
        
        # Simulate clients
        client_gradients = simulate_client_gradients(
            num_clients=4,
            model=model,
            add_anomaly=(round_num == 4)  # Anomaly in round 4
        )
        
        result = server.aggregate_round(
            client_gradients=client_gradients,
            client_ids=[f'hospital_{i}' for i in range(1, 5)],
            round_number=round_num
        )
        
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    for r in results:
        anomaly_status = "⚠️  ANOMALY" if r['anomaly_detected'] else "✅ OK"
        blockchain_status = "✅" if r['blockchain_receipt'] else "❌"
        
        print(f"\nRound {r['round']}: {anomaly_status}")
        print(f"   Clients: {r['num_clients']}")
        print(f"   Blockchain: {blockchain_status}")
        if r['anomaly_score']:
            print(f"   Anomaly score: {r['anomaly_score']:.2f}")
    
    return results


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("BLOCKCHAIN + HOMOMORPHIC ENCRYPTION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Basic Aggregation", test_basic_aggregation),
        ("Blockchain Integration", test_blockchain_integration),
        ("Anomaly Detection", test_anomaly_detection),
        ("Multiple Rounds", test_multiple_rounds)
    ]
    
    print("\nAvailable Tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"   {i}. {name}")
    print(f"   {len(tests)+1}. Run All Tests")
    
    try:
        choice = input("\nSelect test (1-5) or press Enter for all: ").strip()
        
        if not choice or choice == str(len(tests)+1):
            # Run all tests
            for name, test_func in tests:
                try:
                    test_func()
                except Exception as e:
                    print(f"\n❌ {name} failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Run selected test
            idx = int(choice) - 1
            if 0 <= idx < len(tests):
                name, test_func = tests[idx]
                test_func()
            else:
                print("Invalid selection")
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
