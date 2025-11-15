"""
Quick Start Example: Blockchain + Homomorphic Encryption

This is a minimal example showing how to use the blockchain integration.
Run this after starting Hardhat node and deploying the contract.
"""

import numpy as np
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from blockchain_integration import BlockchainFLServer


def main():
    print("\n" + "=" * 70)
    print("QUICK START: BLOCKCHAIN FEDERATED LEARNING")
    print("=" * 70)
    
    # Step 1: Initialize server
    print("\n📋 Step 1: Initialize server...")
    server = BlockchainFLServer(
        enable_blockchain=True,
        enable_anomaly_detection=True,
        anomaly_threshold=3.0
    )
    
    # Step 2: Simulate 3 hospitals with gradient updates
    print("\n📋 Step 2: Simulate hospital gradients...")
    
    # Each hospital has gradients for 3 layers (simplified)
    hospital_1_gradients = [
        np.random.randn(10, 20) * 0.01,  # Layer 1 weights
        np.random.randn(20) * 0.01,      # Layer 1 bias
        np.random.randn(20, 5) * 0.01,   # Layer 2 weights
    ]
    
    hospital_2_gradients = [
        np.random.randn(10, 20) * 0.01,
        np.random.randn(20) * 0.01,
        np.random.randn(20, 5) * 0.01,
    ]
    
    hospital_3_gradients = [
        np.random.randn(10, 20) * 0.01,
        np.random.randn(20) * 0.01,
        np.random.randn(20, 5) * 0.01,
    ]
    
    all_gradients = [hospital_1_gradients, hospital_2_gradients, hospital_3_gradients]
    hospital_ids = ['Hospital_A', 'Hospital_B', 'Hospital_C']
    
    print(f"   ✅ Generated gradients for {len(all_gradients)} hospitals")
    print(f"   ✅ Each has {len(hospital_1_gradients)} layers")
    
    # Step 3: Aggregate with blockchain logging
    print("\n📋 Step 3: Aggregate and log to blockchain...")
    
    result = server.aggregate_round(
        client_gradients=all_gradients,
        client_ids=hospital_ids,
        round_number=1
    )
    
    # Step 4: Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n✅ Aggregation Complete!")
    print(f"   Clients aggregated: {result['num_clients']}")
    print(f"   Gradient hash: {result['gradient_hash'][:32]}...")
    
    if result['blockchain_receipt']:
        print(f"\n✅ Blockchain Logged!")
        print(f"   Transaction: {result['blockchain_receipt']['transactionHash'].hex()}")
        print(f"   Block: {result['blockchain_receipt']['blockNumber']}")
        print(f"   Gas used: {result['blockchain_receipt']['gasUsed']}")
    else:
        print(f"\n  Blockchain logging skipped (node not running)")
    
    print(f"\n✅ Anomaly Detection:")
    print(f"   Anomaly detected: {result['anomaly_detected']}")
    if result['anomaly_score']:
        print(f"   Anomaly score: {result['anomaly_score']:.2f}")
    
    print(f"\n✅ Encryption Stats:")
    he_stats = result['he_stats']
    print(f"   Encryptions: {he_stats['encryptions_performed']}")
    print(f"   Aggregations: {he_stats['aggregations_performed']}")
    print(f"   Decryptions: {he_stats['decryptions_performed']}")
    print(f"   Security: {he_stats['security_level']}")
    
    # Step 5: Retrieve blockchain logs
    if server.blockchain_enabled:
        print("\n📋 Step 5: Retrieve blockchain logs...")
        logs = server.get_blockchain_logs(limit=5)
        
        print(f"\n📊 Recent Blockchain Logs (last {len(logs)}):")
        for log in logs[-3:]:  # Show last 3
            status = "🚩 FLAGGED" if log['flagged'] else "✅ OK"
            print(f"\n   Log #{log['index']} {status}")
            print(f"      Hospital: {log['hospital'][:10]}...")
            print(f"      Hash: {log['gradientHash'][:24]}...")
            print(f"      Timestamp: {log['timestamp']}")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   1. Run multiple rounds to see anomaly detection")
    print("   2. Run test_integration.py for comprehensive tests")
    print("   3. Integrate with your actual PyTorch model")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Make sure:")
        print("   1. Hardhat node is running: npx hardhat node")
        print("   2. Contract is deployed: npx hardhat run scripts/deploy_proxy.js --network localhost")
        print("   3. Dependencies installed: pip install tenseal web3 numpy")
