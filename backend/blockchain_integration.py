"""
Blockchain Integration for Federated Learning with Homomorphic Encryption

This module integrates:
1. Homomorphic Encryption (TenSEAL) for gradient privacy
2. Blockchain (Ethereum) for immutable audit trail
3. Anomaly Detection for malicious gradient detection

Workflow:
1. Client encrypts gradients using HE
2. Server aggregates encrypted gradients (homomorphic operations)
3. Server decrypts aggregated result
4. Hash encrypted gradients and log to blockchain
5. Detect anomalies in aggregated gradients
6. Flag suspicious updates on blockchain
"""

import sys
import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import tenseal as ts

# Add blockchain python directory to path
blockchain_path = Path(__file__).parent.parent / "blockchain" / "python"
sys.path.insert(0, str(blockchain_path))

try:
    from blockchain_client import log_gradient_update, contract, w3
except ImportError as e:
    print(f"⚠️  Warning: blockchain_client not found. Blockchain features disabled.")
    print(f"   Error: {e}")
    print(f"   Path checked: {blockchain_path}")
    log_gradient_update = None
    contract = None
    w3 = None

from utils.encryption import HEManager
from utils.anomaly_detector import GradientAnomalyDetector


class BlockchainFLServer:
    """
    Federated Learning Server with Blockchain Integration
    
    Features:
    - Homomorphic encryption for gradient privacy
    - Blockchain logging for audit trail
    - Anomaly detection for malicious updates
    """
    
    def __init__(
        self,
        enable_blockchain: bool = True,
        enable_anomaly_detection: bool = True,
        anomaly_threshold: float = 3.0
    ):
        """
        Initialize the FL server
        
        Args:
            enable_blockchain: Enable blockchain logging
            enable_anomaly_detection: Enable anomaly detection
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        print("=" * 60)
        print("INITIALIZING BLOCKCHAIN FEDERATED LEARNING SERVER")
        print("=" * 60)
        
        # Initialize Homomorphic Encryption
        self.he_manager = HEManager()
        
        # Initialize Anomaly Detector
        self.anomaly_detector = None
        if enable_anomaly_detection:
            self.anomaly_detector = GradientAnomalyDetector(threshold=anomaly_threshold)
            print(f"✅ Anomaly detector initialized (threshold: {anomaly_threshold})")
        
        # Check blockchain availability
        self.blockchain_enabled = enable_blockchain and (log_gradient_update is not None)
        if self.blockchain_enabled:
            try:
                # Test blockchain connection
                if w3 and w3.is_connected():
                    print(f"✅ Connected to blockchain: {w3.provider.endpoint_uri}")
                    print(f"   Contract address: {contract.address if contract else 'N/A'}")
                else:
                    self.blockchain_enabled = False
                    print("⚠️  Blockchain connection failed. Running without blockchain.")
            except Exception as e:
                self.blockchain_enabled = False
                print(f"⚠️  Blockchain error: {e}. Running without blockchain.")
        else:
            print("ℹ️  Blockchain disabled")
        
        print("=" * 60)
        print()
    
    def hash_encrypted_gradients(self, encrypted_gradients: List) -> str:
        """
        Create a hash of encrypted gradients for blockchain logging
        
        Args:
            encrypted_gradients: List of encrypted CKKS vectors
        
        Returns:
            SHA-256 hash string
        """
        # Serialize encrypted gradients
        serialized = []
        for enc_vec in encrypted_gradients:
            serialized.append(enc_vec.serialize())
        
        # Combine all serialized data
        combined = b''.join(serialized)
        
        # Create hash
        hash_obj = hashlib.sha256(combined)
        return hash_obj.hexdigest()
    
    def log_to_blockchain(self, gradient_hash: str) -> Optional[Dict]:
        """
        Log gradient hash to blockchain
        
        Args:
            gradient_hash: SHA-256 hash of encrypted gradients
        
        Returns:
            Transaction receipt or None if blockchain disabled
        """
        if not self.blockchain_enabled:
            print("⚠️  Blockchain disabled, skipping logging")
            return None
        
        try:
            print(f"📝 Logging to blockchain...")
            print(f"   Hash: {gradient_hash[:16]}...")
            
            receipt = log_gradient_update(gradient_hash)
            
            print(f"✅ Logged to blockchain")
            print(f"   Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"   Block number: {receipt['blockNumber']}")
            print(f"   Gas used: {receipt['gasUsed']}")
            print()
            
            return receipt
        except Exception as e:
            print(f"❌ Blockchain logging failed: {e}")
            return None
    
    def flag_update_on_blockchain(self, update_index: int) -> Optional[Dict]:
        """
        Flag a suspicious update on the blockchain
        
        Args:
            update_index: Index of the update to flag
        
        Returns:
            Transaction receipt or None
        """
        if not self.blockchain_enabled or not contract:
            print("⚠️  Blockchain disabled, cannot flag update")
            return None
        
        try:
            print(f"🚩 Flagging update {update_index} on blockchain...")
            
            # Build transaction
            account = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
            private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            
            nonce = w3.eth.get_transaction_count(account)
            tx = contract.functions.flagUpdate(update_index).build_transaction({
                "from": account,
                "nonce": nonce,
                "gas": 500000,
                "gasPrice": w3.to_wei("1", "gwei"),
            })
            
            # Sign and send
            signed_tx = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"✅ Update flagged on blockchain")
            print(f"   Transaction hash: {receipt['transactionHash'].hex()}")
            print()
            
            return receipt
        except Exception as e:
            print(f"❌ Failed to flag update: {e}")
            return None
    
    def aggregate_round(
        self,
        client_gradients: List[List[np.ndarray]],
        client_ids: Optional[List[str]] = None,
        round_number: int = 0
    ) -> Dict:
        """
        Perform one round of federated aggregation with blockchain logging
        
        Args:
            client_gradients: List of gradient lists (one per client)
            client_ids: Optional list of client identifiers
            round_number: Training round number
        
        Returns:
            Dictionary with aggregation results and metadata
        """
        print("\n" + "=" * 60)
        print(f"FEDERATED LEARNING ROUND {round_number}")
        print("=" * 60)
        print(f"Number of clients: {len(client_gradients)}")
        print(f"Layers per client: {len(client_gradients[0]) if client_gradients else 0}")
        print()
        
        if not client_gradients:
            raise ValueError("No client gradients provided")
        
        # Generate client IDs if not provided
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(client_gradients))]
        
        # Step 1: Check if gradients are already encrypted or need encryption
        print("STEP 1: PREPARING CLIENT GRADIENTS")
        print("-" * 60)
        encrypted_list = []
        for i, (client_id, gradients) in enumerate(zip(client_ids, client_gradients)):
            print(f"Client {i+1} ({client_id}):")
            # Check if gradients are already encrypted (CKKSVector objects)
            if gradients and isinstance(gradients[0], ts.CKKSVector):
                print(f"   ✓ Already encrypted: {len(gradients)} layers")
                encrypted_list.append(gradients)
            else:
                # Need to encrypt
                encrypted = self.he_manager.encrypt_gradients(gradients, encrypt_all=False)
                encrypted_list.append(encrypted)
        
        # Step 2: Homomorphic aggregation
        print("\nSTEP 2: HOMOMORPHIC AGGREGATION")
        print("-" * 60)
        aggregated_encrypted = self.he_manager.aggregate_encrypted(encrypted_list)
        
        # Step 3: Hash encrypted gradients for blockchain
        print("\nSTEP 3: BLOCKCHAIN LOGGING")
        print("-" * 60)
        gradient_hash = self.hash_encrypted_gradients(aggregated_encrypted)
        blockchain_receipt = self.log_to_blockchain(gradient_hash)
        
        # Step 4: Decrypt aggregated result
        print("\nSTEP 4: DECRYPTING AGGREGATED RESULT")
        print("-" * 60)
        aggregated_gradients = self.he_manager.decrypt_gradients(aggregated_encrypted)
        
        # Step 5: Anomaly detection
        print("\nSTEP 5: ANOMALY DETECTION")
        print("-" * 60)
        anomaly_detected = False
        anomaly_score = None
        
        if self.anomaly_detector:
            # Use the aggregated gradients for anomaly detection
            client_id = client_ids[0] if client_ids else "aggregated"
            
            # Check for anomalies using the correct API
            is_anomaly, z_score, reason = self.anomaly_detector.check_update(
                aggregated_gradients,
                hospital_id=client_id,
                round_num=round_number
            )
            anomaly_score = z_score
            
            if is_anomaly:
                anomaly_detected = True
                
                # Flag on blockchain if available
                if self.blockchain_enabled and blockchain_receipt:
                    # Get the log count to flag the latest update
                    try:
                        log_count = contract.functions.getLogsCount().call()
                        self.flag_update_on_blockchain(log_count - 1)
                    except Exception as e:
                        print(f"❌ Failed to get log count: {e}")
            else:
                print(f"✅ No anomaly detected. Z-score: {anomaly_score:.2f}")
        
        print("\n" + "=" * 60)
        print("ROUND COMPLETE")
        print("=" * 60)
        print()
        
        return {
            'round': round_number,
            'num_clients': len(client_gradients),
            'aggregated_gradients': aggregated_gradients,
            'gradient_hash': gradient_hash,
            'blockchain_receipt': blockchain_receipt,
            'anomaly_detected': anomaly_detected,
            'anomaly_score': anomaly_score,
            'he_stats': self.he_manager.get_context_info()
        }
    
    def get_blockchain_logs(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent logs from blockchain
        
        Args:
            limit: Maximum number of logs to retrieve
        
        Returns:
            List of log entries
        """
        if not self.blockchain_enabled or not contract:
            print("⚠️  Blockchain disabled")
            return []
        
        try:
            log_count = contract.functions.getLogsCount().call()
            print(f"📊 Total logs on blockchain: {log_count}")
            
            logs = []
            start_idx = max(0, log_count - limit)
            
            for i in range(start_idx, log_count):
                log_entry = contract.functions.logs(i).call()
                logs.append({
                    'index': i,
                    'hospital': log_entry[0],
                    'gradientHash': log_entry[1],
                    'timestamp': log_entry[2],
                    'flagged': log_entry[3]
                })
            
            return logs
        except Exception as e:
            print(f"❌ Failed to retrieve logs: {e}")
            return []
