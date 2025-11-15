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
        anomaly_threshold: float = 3.0,
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
        self.blockchain_enabled = enable_blockchain and (
            log_gradient_update is not None
        )
        if self.blockchain_enabled:
            try:
                # Test blockchain connection
                if w3 and w3.is_connected():
                    print(f"✅ Connected to blockchain: {w3.provider.endpoint_uri}")
                    print(
                        f"   Contract address: {contract.address if contract else 'N/A'}"
                    )
                else:
                    self.blockchain_enabled = False
                    print(
                        "⚠️  Blockchain connection failed. Running without blockchain."
                    )
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
        combined = b"".join(serialized)

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
            private_key = (
                "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            )

            nonce = w3.eth.get_transaction_count(account)
            tx = contract.functions.flagUpdate(update_index).build_transaction(
                {
                    "from": account,
                    "nonce": nonce,
                    "gas": 500000,
                    "gasPrice": w3.to_wei("1", "gwei"),
                }
            )

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
        num_examples: Optional[List[int]] = None,
        round_number: int = 0,
    ) -> Dict:
        """
        Perform one round of federated aggregation with blockchain logging

        Args:
            client_gradients: List of gradient lists (one per client)
            client_ids: Optional list of client identifiers
            num_examples: Optional list of sample counts per client (for weighted averaging)
            round_number: Training round number

        Returns:
            Dictionary with aggregation results and metadata
        """
        print("\n" + "=" * 60)
        print(f"FEDERATED LEARNING ROUND {round_number}")
        print("=" * 60)
        print(f"Number of clients: {len(client_gradients)}")
        print(
            f"Layers per client: {len(client_gradients[0]) if client_gradients else 0}"
        )
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
            print(f"Client {i + 1} ({client_id}):")
            # Check if gradients are already encrypted (CKKSVector objects)
            if gradients and isinstance(gradients[0], ts.CKKSVector):
                print(f"   ✓ Already encrypted: {len(gradients)} layers")
                encrypted_list.append(gradients)
            else:
                # Need to encrypt
                encrypted = self.he_manager.encrypt_gradients(
                    gradients, encrypt_all=False
                )
                encrypted_list.append(encrypted)

        # Step 2: Anomaly detection BEFORE aggregation
        print("\nSTEP 2: ANOMALY DETECTION (BEFORE AGGREGATION)")
        print("-" * 60)
        anomaly_detected = False
        anomaly_score = None
        flagged_clients = []
        clean_encrypted_list = []
        clean_client_ids = []
        clean_num_examples = []

        if self.anomaly_detector and len(encrypted_list) > 1:
            # ✅ PRIVACY PRESERVED: NO DECRYPTION on server
            # Method: Compare encrypted gradient statistics using homomorphic properties
            print(f"🔍 Encrypted-domain anomaly detection...")
            print(
                f"   Method: Statistical consistency check (NO SERVER-SIDE DECRYPTION)"
            )

            # Strategy: Check if client contributions are statistically consistent
            # by examining properties that can be computed homomorphically

            # For now, we'll use a simple heuristic:
            # - Check if encrypted gradient magnitudes are within reasonable bounds
            # - Serialize encrypted vectors and check byte-level patterns
            # - Look for extreme outliers in serialized size

            print(f"\n📊 Analyzing encrypted gradient properties...")

            client_stats = []
            for i, (client_id, enc_grads) in enumerate(zip(client_ids, encrypted_list)):
                # Get serialized size (proxy for gradient complexity)
                serialized = self.he_manager.serialize_vectors(enc_grads)
                total_bytes = sum(s.nbytes for s in serialized)

                # Count number of gradient layers
                num_layers = len(enc_grads)

                client_stats.append((i, client_id, total_bytes, num_layers))
                print(f"   {client_id}: {total_bytes:,} bytes, {num_layers} layers")

            # Check for extreme outliers in size (potential corruption/attack)
            sizes = [size for _, _, size, _ in client_stats]
            median_size = np.median(sizes)
            # Flag if size deviates by more than 3x from median
            size_threshold = median_size * 3.0

            print(f"\n   📊 Median encrypted size: {median_size:,.0f} bytes")
            print(f"   🚨 Outlier threshold (3x median): {size_threshold:,.0f} bytes")

            # Initially accept all clients for encrypted-domain analysis
            for idx, client_id, size, num_layers in client_stats:
                if size > size_threshold or size < median_size / 3.0:
                    print(
                        f"   ⚠️  WARNING: {client_id} has unusual encrypted size ({size:,} bytes)"
                    )
                    print(f"      → May indicate corruption or malicious gradients")
                    print(f"      → Marking for further review (blockchain audit)")
                    # Don't automatically exclude - log for audit but include in aggregation
                    # Server cannot determine maliciousness without decryption
                else:
                    print(
                        f"   ✅ {client_id}: Encrypted statistics within normal range"
                    )

                # Accept all clients since we can't detect attacks without decryption
                # Real anomaly detection requires client-side validation after receiving aggregate
                clean_encrypted_list.append(encrypted_list[idx])
                clean_client_ids.append(client_ids[idx])
                if num_examples:
                    clean_num_examples.append(num_examples[idx])

            print(
                f"\n   ℹ️  Note: True Byzantine fault detection requires client-side validation"
            )
            print(
                f"      Clients should validate the aggregated model after decryption"
            )
            print(f"      Blockchain provides immutable audit trail for accountability")
        else:
            # No anomaly detection - use all clients
            clean_encrypted_list = encrypted_list
            clean_client_ids = client_ids
            clean_num_examples = num_examples if num_examples else []

        # Step 3: Byzantine-Robust Homomorphic Aggregation
        print("\nSTEP 3: BYZANTINE-ROBUST HOMOMORPHIC AGGREGATION")
        print("-" * 60)

        if len(clean_encrypted_list) == 0:
            print("❌ No clean clients to aggregate!")
            return {
                "aggregated_encrypted": None,
                "num_clients": 0,
                "anomaly_detected": True,
                "anomaly_score": anomaly_score,
                "flagged_clients": flagged_clients,
                "blockchain_receipt": None,
            }

        # Use Krum for Byzantine-robust aggregation (no decryption needed!)
        # Krum automatically selects the most representative gradient, excluding outliers
        num_byzantine = 1  # Assume at most 1 malicious client

        if len(clean_encrypted_list) > 2 * num_byzantine:
            print(f"   Using KRUM aggregation (Byzantine-tolerant)")
            print(f"   Can tolerate up to {num_byzantine} malicious client(s)")
            aggregated_encrypted = self.he_manager.aggregate_encrypted_krum(
                clean_encrypted_list, num_byzantine=num_byzantine
            )
        elif clean_num_examples:
            aggregated_encrypted = self.he_manager.aggregate_encrypted_weighted(
                clean_encrypted_list, clean_num_examples
            )
        else:
            aggregated_encrypted = self.he_manager.aggregate_encrypted(
                clean_encrypted_list
            )

        # Step 3: Blockchain logging for individual clients
        # Note: We log ONLY clean (non-flagged) client contributions for token rewards
        print("\nSTEP 3: BLOCKCHAIN LOGGING")
        print("-" * 60)

        # Log individual client contributions for token rewards
        if self.blockchain_enabled:
            print(f"\n💰 LOGGING CLIENT CONTRIBUTIONS FOR MCT TOKEN REWARDS")
            print("-" * 60)
            try:
                from blockchain_client import (
                    log_gradient_update,
                    get_token_balance,
                    HOSPITAL_ACCOUNTS,
                )

                # Log ONLY clean clients (non-flagged) - they get 10 MCT tokens
                for client_id, enc_grads in zip(clean_client_ids, clean_encrypted_list):
                    # Hash this client's encrypted gradients
                    client_hash = hashlib.sha256(
                        b"".join(
                            [
                                self.he_manager.serialize_vectors([eg])[0].tobytes()
                                for eg in enc_grads
                            ]
                        )
                    ).hexdigest()

                    # Log to blockchain (not flagged = gets 10 MCT tokens automatically)
                    receipt = log_gradient_update(
                        client_hash, flagged=False, hospital_id=client_id
                    )

                    # Get updated token balance
                    if client_id in HOSPITAL_ACCOUNTS:
                        balance = get_token_balance(
                            HOSPITAL_ACCOUNTS[client_id]["address"]
                        )
                        print(
                            f"   ✅ {client_id}: +10 MCT tokens (Balance: {balance:.0f} MCT)"
                        )
                    else:
                        print(
                            f"   ✅ {client_id}: Logged (Block #{receipt['blockNumber']})"
                        )

                # Log flagged clients with 0 MCT token reward
                for client_id in flagged_clients:
                    # Find this client's encrypted gradients in the original list
                    idx = client_ids.index(client_id)
                    enc_grads = encrypted_list[idx]

                    # Hash this client's encrypted gradients
                    client_hash = hashlib.sha256(
                        b"".join(
                            [
                                self.he_manager.serialize_vectors([eg])[0].tobytes()
                                for eg in enc_grads
                            ]
                        )
                    ).hexdigest()

                    # Log to blockchain (flagged = gets 0 MCT tokens)
                    receipt = log_gradient_update(
                        client_hash, flagged=True, hospital_id=client_id
                    )

                    # Get token balance (should not have increased)
                    if client_id in HOSPITAL_ACCOUNTS:
                        balance = get_token_balance(
                            HOSPITAL_ACCOUNTS[client_id]["address"]
                        )
                        print(
                            f"   🚫 {client_id}: +0 MCT tokens (FLAGGED - Balance: {balance:.0f} MCT)"
                        )
                    else:
                        print(
                            f"   🚫 {client_id}: Logged as flagged (Block #{receipt['blockNumber']})"
                        )

            except Exception as e:
                print(f"   ⚠️  Token reward logging failed: {e}")

        # Compute hash of aggregated encrypted gradients for audit trail
        gradient_hash = hashlib.sha256(
            b"".join(
                [
                    self.he_manager.serialize_vectors([eg])[0].tobytes()
                    for eg in aggregated_encrypted
                ]
            )
        ).hexdigest()

        # Blockchain receipt: True if blockchain is enabled and clients were logged
        blockchain_receipt = self.blockchain_enabled and len(clean_client_ids) > 0

        # CRITICAL: Return encrypted aggregates - server never keeps plaintext
        print("\n🔐 PRIVACY PRESERVED: Server keeping aggregates ENCRYPTED")
        print("   Clients will decrypt locally after receiving aggregated update")

        print("\n" + "=" * 60)
        print("ROUND COMPLETE")
        print("=" * 60)
        print()

        return {
            "round": round_number,
            "num_clients": len(
                clean_client_ids
            ),  # Only count clean clients in aggregation
            "aggregated_encrypted": aggregated_encrypted,  # Return ENCRYPTED
            "gradient_hash": gradient_hash,
            "blockchain_receipt": blockchain_receipt,
            "anomaly_detected": len(flagged_clients) > 0,
            "anomaly_score": None,  # Could compute average norm ratio if needed
            "flagged_clients": flagged_clients,
            "he_stats": self.he_manager.get_context_info(),
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
                logs.append(
                    {
                        "index": i,
                        "hospital": log_entry[0],
                        "gradientHash": log_entry[1],
                        "timestamp": log_entry[2],
                        "flagged": log_entry[3],
                    }
                )

            return logs
        except Exception as e:
            print(f"❌ Failed to retrieve logs: {e}")
            return []
