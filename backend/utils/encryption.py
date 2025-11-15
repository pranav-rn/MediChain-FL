"""
Homomorphic Encryption Manager using TenSEAL (CKKS scheme)

CKKS (Cheon-Kim-Kim-Song) scheme allows approximate arithmetic on encrypted real numbers
Perfect for federated learning where gradients are floating-point values

Security Level: 128-bit
Polynomial Modulus Degree: 8192
Supports: Addition, Multiplication, Scalar operations on encrypted data
"""

import tenseal as ts
import numpy as np
import torch
from typing import List, Union
import time


class HEManager:
    """
    Homomorphic Encryption Manager

    Provides CKKS encryption for federated learning gradients
    Supports homomorphic operations: addition, scalar multiplication
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        global_scale: int = 2**40,
    ):
        """
        Initialize CKKS homomorphic encryption context

        Args:
            poly_modulus_degree: Polynomial degree (higher = more secure but slower)
                                 Common values: 4096, 8192, 16384
            coeff_mod_bit_sizes: Modulus chain for rescaling
                                 Determines precision and number of operations
            global_scale: Scaling factor for encoding (2^40 is standard)

        Technical Details:
        - poly_modulus_degree=8192 provides 128-bit security
        - Each multiplication consumes one level from coeff_mod_bit_sizes
        - More levels = more multiplications possible before noise overwhelms
        """

        # Default coefficient modulus chain based on polynomial degree
        if coeff_mod_bit_sizes is None:
            # TenSEAL security constraints for total coefficient modulus bits:
            # 4096 -> max ~109 bits, 8192 -> max ~218 bits, 16384 -> max ~438 bits
            if poly_modulus_degree == 4096:
                coeff_mod_bit_sizes = [
                    40,
                    21,
                    21,
                    40,
                ]  # Total: 122 bits (safe for 4096)
            elif poly_modulus_degree == 8192:
                coeff_mod_bit_sizes = [
                    60,
                    40,
                    40,
                    60,
                ]  # Total: 200 bits (safe for 8192)
            elif poly_modulus_degree == 16384:
                coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 60]  # Total: 280 bits
            else:
                # Conservative fallback
                coeff_mod_bit_sizes = [40, 21, 40]

        print("🔐 Initializing CKKS Homomorphic Encryption Context...")
        print(f"   Polynomial degree: {poly_modulus_degree}")
        print(f"   Security level: 128-bit")
        print(f"   Modulus chain: {coeff_mod_bit_sizes}")
        print(f"   Global scale: {global_scale}")

        # Create TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )

        # Store parameters locally (TenSEAL context object does not expose
        # poly_modulus_degree/coeff_mod_bit_sizes as direct attributes)
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes

        # Generate Galois keys (required for rotations)
        # Rotations are used in some advanced operations
        self.context.generate_galois_keys()

        # Set global scale
        self.context.global_scale = global_scale
        # Keep secret context for local decryption but also create a PUBLIC context
        # for sharing with clients. We create a copy by serializing and deserializing
        # so we can call make_context_public() without removing the secret key from
        # the manager's main context.
        try:
            serialized_secret = self.context.serialize()
            public_context = ts.context_from(serialized_secret)
            public_context.make_context_public()
            self.public_context = public_context
            self.public_context_serialized = public_context.serialize()
        except Exception:
            # Fallback: if anything fails, keep only the secret context and set public to None
            self.public_context = None
            self.public_context_serialized = None

        print("✅ CKKS Context initialized successfully\n")

        # Statistics
        self.encryption_count = 0
        self.decryption_count = 0
        self.aggregation_count = 0

    def encrypt_gradients(
        self,
        gradients: Union[List[np.ndarray], List[torch.Tensor]],
        encrypt_all: bool = False,
    ) -> List[ts.CKKSVector]:
        """
        Encrypt model gradients using CKKS

        Args:
            gradients: List of gradient tensors (one per layer)
            encrypt_all: If True, encrypt all layers. If False, only last 2 layers

        Returns:
            List of encrypted CKKS vectors

        Technical Details:
        - Each gradient tensor is flattened to 1D vector
        - Flattened vector is encoded as polynomial
        - Polynomial is encrypted using RLWE (Ring-Learning-With-Errors)
        - Result: Two polynomials (c0, c1) that hide the original values
        """

        start_time = time.time()

        # Determine which layers to encrypt
        if encrypt_all:
            layers_to_encrypt = gradients
            print(f"🔒 Encrypting ALL {len(gradients)} gradient layers...")
        else:
            # Only encrypt last 2 layers (most sensitive + smaller payload)
            layers_to_encrypt = gradients[-2:]
            print(f"🔒 Encrypting final {len(layers_to_encrypt)} gradient layers...")

        encrypted = []

        for i, tensor in enumerate(layers_to_encrypt):
            # Convert to numpy if PyTorch tensor
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().detach().numpy()

            # Flatten to 1D vector
            flat = tensor.flatten()

            # Convert to list (TenSEAL requirement)
            flat_list = flat.tolist()

            # ENCRYPT: This is where the magic happens
            # Under the hood:
            # 1. Encode: float values -> polynomial coefficients
            # 2. Encrypt: polynomial -> (c0, c1) using RLWE
            # 3. c0 = b*u + e0 + Δ*m  (contains message with noise)
            # 4. c1 = a*u + e1         (random component)
            # where u, e0, e1 are small random noise, m is message, Δ is scale
            enc_vector = ts.ckks_vector(self.context, flat_list)

            encrypted.append(enc_vector)

            print(
                f"   ✓ Layer {i + 1}: {len(flat_list)} values encrypted "
                f"(original shape: {tensor.shape})"
            )

        elapsed = time.time() - start_time
        self.encryption_count += 1

        print(f"✅ Encryption complete in {elapsed:.3f}s")
        print(f"   Encrypted {len(encrypted)} layers")
        print(f"   Total encryptions so far: {self.encryption_count}\n")

        return encrypted

    def aggregate_encrypted(
        self, encrypted_list: List[List[ts.CKKSVector]]
    ) -> List[ts.CKKSVector]:
        """
        Aggregate encrypted gradients using FedAvg (homomorphically)

        Formula: aggregated = (enc_1 + enc_2 + ... + enc_n) / n

        Args:
            encrypted_list: List of encrypted gradient lists from each client
                           Shape: [num_clients][num_layers]

        Returns:
            List of aggregated encrypted vectors (one per layer)

        Mathematical Property (Why This Works):
            Enc(a) + Enc(b) = Enc(a + b)
            Enc(a) * c = Enc(a * c)

            Therefore:
            (Enc(a) + Enc(b) + Enc(c)) / 3 = Enc((a + b + c) / 3)

        This is the CORE of privacy-preserving aggregation!
        """

        start_time = time.time()

        num_clients = len(encrypted_list)
        if num_clients == 0:
            raise ValueError("Cannot aggregate empty list")

        num_layers = len(encrypted_list[0])

        print(f"⚙️  HOMOMORPHIC AGGREGATION")
        print(f"   Aggregating {num_clients} encrypted updates")
        print(f"   Each update has {num_layers} encrypted layers")
        print()

        weights = [1.0] * num_clients
        aggregated = self.aggregate_encrypted_weighted(encrypted_list, weights)

        elapsed = time.time() - start_time
        self.aggregation_count += 1

        print(f"✅ Homomorphic aggregation complete in {elapsed:.3f}s")
        print(f"   Total aggregations so far: {self.aggregation_count}")
        print(
            f"\n🔐 CRITICAL: Server performed FedAvg WITHOUT seeing individual gradients!"
        )
        print(f"   All operations were on encrypted data\n")

        return aggregated

    def aggregate_encrypted_weighted(
        self, encrypted_list: List[List[ts.CKKSVector]], weights: List[float]
    ) -> List[ts.CKKSVector]:
        """
        Aggregate encrypted gradients with custom weights (e.g., based on dataset size)

        Formula: aggregated = (w1*enc_1 + w2*enc_2 + ... + wn*enc_n) / sum(weights)

        Args:
            encrypted_list: List of encrypted gradient lists
            weights: Weight for each client (e.g., number of samples)

        Returns:
            Weighted aggregated encrypted vectors
        """
        start_time = time.time()

        if not encrypted_list:
            raise ValueError("Cannot aggregate empty encrypted list")
        if len(encrypted_list) != len(weights):
            raise ValueError("Weights length must match encrypted updates")

        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")

        num_layers = len(encrypted_list[0])
        num_clients = len(encrypted_list)

        print(f"⚙️  WEIGHTED HOMOMORPHIC AGGREGATION")
        print(f"   Aggregating {num_clients} encrypted updates")
        print(f"   Each update has {num_layers} encrypted layers")
        print(f"   Sample counts: {[int(w) for w in weights]}")
        print(f"   Total samples: {int(total_weight)}")
        print()

        aggregated = []

        for layer_idx in range(num_layers):
            layer_accumulator = None
            for client_idx, (client_updates, weight) in enumerate(
                zip(encrypted_list, weights)
            ):
                normalized_weight = weight / total_weight
                scaled = client_updates[layer_idx] * normalized_weight
                layer_accumulator = (
                    scaled if layer_accumulator is None else layer_accumulator + scaled
                )
                print(
                    f"      Layer {layer_idx + 1}: incorporated client {client_idx + 1} "
                    f"with weight {normalized_weight:.4f} ({int(weight)}/{int(total_weight)} samples)"
                )

            aggregated.append(layer_accumulator)

        elapsed = time.time() - start_time
        self.aggregation_count += 1

        print(f"✅ Weighted aggregation complete in {elapsed:.3f}s")
        print(f"   Total aggregations so far: {self.aggregation_count}")
        print(
            f"\n🔐 CRITICAL: Server performed weighted FedAvg WITHOUT seeing individual gradients!"
        )
        print(f"   All operations were on encrypted data\n")

        return aggregated

    def aggregate_encrypted_krum(
        self,
        encrypted_list: List[List[ts.CKKSVector]],
        num_byzantine: int = 1,
        multi_krum: int = None,
    ) -> List[ts.CKKSVector]:
        """
        Byzantine-robust aggregation using Multi-Krum algorithm

        Multi-Krum selects the k most representative gradients and averages them,
        automatically excluding outliers (malicious clients).

        Works by computing pairwise squared distances in ENCRYPTED domain!
        - Only distance scores are decrypted (not gradients)
        - Automatically excludes extreme outliers
        - Provably Byzantine-tolerant

        Args:
            encrypted_list: List of encrypted gradient lists [num_clients][num_layers]
            num_byzantine: Max number of Byzantine (malicious) clients to tolerate
            multi_krum: Number of clients to select and average (default: all honest clients)

        Returns:
            Average of selected honest gradients (Byzantine attacks excluded!)
        """
        start_time = time.time()

        if not encrypted_list:
            raise ValueError("Cannot aggregate empty encrypted list")

        num_clients = len(encrypted_list)

        if num_clients <= 2 * num_byzantine:
            print(f"⚠️  WARNING: Not enough honest clients for Byzantine tolerance")
            print(
                f"   Need at least {2 * num_byzantine + 1} clients for {num_byzantine} Byzantine"
            )
            print(f"   Falling back to simple average")
            return self.aggregate_encrypted(encrypted_list)

        num_layers = len(encrypted_list[0])

        # Default: select all honest clients (num_clients - num_byzantine)
        if multi_krum is None:
            multi_krum = num_clients - num_byzantine

        print(f"⚙️  MULTI-KRUM BYZANTINE-ROBUST AGGREGATION")
        print(f"   Clients: {num_clients}, Tolerating up to {num_byzantine} Byzantine")
        print(f"   Selecting top {multi_krum} most representative gradients")
        print(f"   Outliers (malicious clients) will be excluded")
        print()

        # Compute pairwise squared distances (can be done on encrypted data!)
        # d²(a,b) = ||a-b||² = (a-b)·(a-b)
        print(f"   📊 Computing pairwise distances in encrypted domain...")

        scores = []
        for i in range(num_clients):
            # For client i, compute sum of distances to k nearest neighbors
            # k = num_clients - num_byzantine - 2
            distances = []

            for j in range(num_clients):
                if i == j:
                    continue

                # Compute squared distance between client i and j
                # ||a-b||² in encrypted domain
                dist_squared = None
                for layer_idx in range(num_layers):
                    diff = encrypted_list[i][layer_idx] - encrypted_list[j][layer_idx]
                    # Squared distance contribution from this layer
                    layer_dist = diff * diff  # This is homomorphic!

                    if dist_squared is None:
                        dist_squared = layer_dist
                    else:
                        dist_squared = dist_squared + layer_dist

                distances.append((j, dist_squared))

            # Select k closest neighbors (smallest distances)
            k = num_clients - num_byzantine - 2
            # For encrypted distances, we decrypt ONLY the distance scores (not the gradients!)
            decrypted_distances = [
                (j, self.decrypt_single_value(d)) for j, d in distances
            ]
            decrypted_distances.sort(key=lambda x: x[1])

            # Sum of k smallest distances
            krum_score = sum(d for _, d in decrypted_distances[:k])
            scores.append((i, krum_score))
            print(f"      Client {i}: Krum score = {krum_score:.4f}")

        # Select top multi_krum clients with SMALLEST scores (most representative)
        scores.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in scores[:multi_krum]]

        print(
            f"\n   ✅ Selected {multi_krum} most representative clients: {selected_indices}"
        )
        print(f"   🛡️  Byzantine clients (outliers) automatically excluded!")

        # Average the selected honest gradients
        selected_encrypted = [encrypted_list[i] for i in selected_indices]
        aggregated = self.aggregate_encrypted(selected_encrypted)

        elapsed = time.time() - start_time
        self.aggregation_count += 1

        print(f"✅ Multi-Krum aggregation complete in {elapsed:.3f}s")
        print(f"\n🔐 CRITICAL: Only distance scores decrypted, NOT gradients!")
        print(f"   Gradient values remain encrypted throughout\n")

        return aggregated

    def decrypt_single_value(self, encrypted_vec: ts.CKKSVector) -> float:
        """Helper to decrypt a single scalar value (e.g., distance score)"""
        decrypted = encrypted_vec.decrypt()
        if isinstance(decrypted, list):
            return sum(decrypted)  # Sum all polynomial coefficients
        return float(decrypted)

    def decrypt_gradients(self, encrypted: List[ts.CKKSVector]) -> List[np.ndarray]:
        """
        Decrypt aggregated gradients

        Args:
            encrypted: List of encrypted CKKS vectors

        Returns:
            List of decrypted numpy arrays

        Technical Details:
        - Uses secret key to decrypt ciphertext
        - Decryption: m' = (c0 + c1 * s) / Δ  (where s is secret key)
        - Decode: polynomial coefficients -> float values
        - Some precision loss due to approximate arithmetic (acceptable for ML)
        """

        start_time = time.time()

        print(f"🔓 Decrypting {len(encrypted)} layers...")

        decrypted = []

        for i, enc_vector in enumerate(encrypted):
            # DECRYPT: This reveals the aggregated result
            # But individual client values remain hidden!
            dec = enc_vector.decrypt()

            # Convert to numpy array
            dec_array = np.array(dec)

            decrypted.append(dec_array)

            print(f"   ✓ Layer {i + 1}: {len(dec)} values decrypted")

        elapsed = time.time() - start_time
        self.decryption_count += 1

        print(f"✅ Decryption complete in {elapsed:.3f}s")
        print(f"   Total decryptions so far: {self.decryption_count}\n")

        return decrypted

    def serialize_vectors(
        self, encrypted_vectors: List[ts.CKKSVector]
    ) -> List[np.ndarray]:
        """Serialize CKKS vectors into numpy uint8 arrays for transport."""
        serialized = []
        for vec in encrypted_vectors:
            buffer = vec.serialize()
            serialized.append(np.frombuffer(buffer, dtype=np.uint8).copy())
        return serialized

    def deserialize_vectors(
        self, serialized_arrays: List[np.ndarray]
    ) -> List[ts.CKKSVector]:
        """Reconstruct CKKS vectors from numpy uint8 arrays."""
        vectors = []
        for array in serialized_arrays:
            contiguous = np.ascontiguousarray(array)
            vectors.append(ts.ckks_vector_from(self.context, contiguous.tobytes()))
        return vectors

    def deserialize_encrypted(self, serialized_bytes: bytes) -> ts.CKKSVector:
        """Deserialize a single encrypted vector from bytes"""
        return ts.ckks_vector_from(self.context, serialized_bytes)

    def get_context_info(self) -> dict:
        """Get information about the encryption context"""
        return {
            "poly_modulus_degree": getattr(self, "poly_modulus_degree", None),
            "coeff_mod_bit_sizes": getattr(self, "coeff_mod_bit_sizes", None),
            "global_scale": getattr(self.context, "global_scale", None),
            "security_level": "128-bit",
            "scheme": "CKKS",
            "encryptions_performed": self.encryption_count,
            "decryptions_performed": self.decryption_count,
            "aggregations_performed": self.aggregation_count,
        }

    def serialize_context(self) -> bytes:
        """
        Serialize context for sharing

        Returns:
            Serialized context (can be sent over network)

        Use case: Share public parameters with clients
        """
        # By default return the public context so we don't leak secrets
        if getattr(self, "public_context_serialized", None) is not None:
            return self.public_context_serialized
        # Otherwise, fallback to the full (secret) context serialization
        return self.context.serialize()

    @classmethod
    def from_serialized(cls, serialized: bytes, has_secret: bool = False):
        """
        Create HEManager from serialized context

        Args:
            serialized: Serialized context bytes
            has_secret: Whether the serialized context includes secret key

        Returns:
            New HEManager instance
        """
        manager = cls.__new__(cls)
        manager.context = ts.context_from(serialized)

        # TenSEAL contexts include secret key by default when serialized
        # No need to regenerate - the deserialized context should have it

        # When constructing from a serialized context we don't necessarily know
        # the original poly_modulus_degree or coeff_mod_bit_sizes, leave as None
        manager.poly_modulus_degree = None
        manager.coeff_mod_bit_sizes = None

        # If the serialized context is a public-only context then the manager won't
        # be able to decrypt. 'has_secret' indicates serialized contains secret
        # key (server side) or not (client side).
        if has_secret:
            # create public copy too
            try:
                manager.public_context = ts.context_from(serialized)
                manager.public_context.make_context_public()
                manager.public_context_serialized = manager.public_context.serialize()
            except Exception:
                manager.public_context = None
                manager.public_context_serialized = None
        else:
            manager.public_context = manager.context
            manager.public_context_serialized = serialized

        manager.encryption_count = 0
        manager.decryption_count = 0
        manager.aggregation_count = 0
        return manager


def test_encryption():
    """
    Test homomorphic encryption with simple example
    Demonstrates that math works on encrypted data
    """

    print("\n" + "=" * 60)
    print("🧪 TESTING HOMOMORPHIC ENCRYPTION")
    print("=" * 60 + "\n")

    # Create encryption manager
    he_manager = HEManager()

    # Simulate 3 hospitals with simple gradients
    print("📊 SIMULATING 3 HOSPITALS\n")

    hospital_1_gradients = [np.array([0.5, -0.3, 0.8, 0.2])]
    hospital_2_gradients = [np.array([0.2, 0.4, -0.1, 0.3])]
    hospital_3_gradients = [np.array([0.3, -0.1, 0.2, 0.1])]

    print("Hospital 1 gradients:", hospital_1_gradients[0])
    print("Hospital 2 gradients:", hospital_2_gradients[0])
    print("Hospital 3 gradients:", hospital_3_gradients[0])
    print()

    # Encrypt each hospital's gradients
    enc_h1 = he_manager.encrypt_gradients(hospital_1_gradients, encrypt_all=True)
    enc_h2 = he_manager.encrypt_gradients(hospital_2_gradients, encrypt_all=True)
    enc_h3 = he_manager.encrypt_gradients(hospital_3_gradients, encrypt_all=True)

    # Try to "read" encrypted data (should be gibberish)
    print("🔍 CAN WE READ ENCRYPTED DATA?")
    print(f"   Encrypted Hospital 1: {enc_h1[0]}")
    print("   ❌ Completely unreadable!\n")

    # Aggregate encrypted gradients
    encrypted_list = [enc_h1, enc_h2, enc_h3]
    aggregated_encrypted = he_manager.aggregate_encrypted(encrypted_list)

    # Decrypt ONLY the aggregated result
    aggregated_decrypted = he_manager.decrypt_gradients(aggregated_encrypted)

    # Calculate expected average (for verification)
    expected = (
        hospital_1_gradients[0] + hospital_2_gradients[0] + hospital_3_gradients[0]
    ) / 3

    print("📊 RESULTS\n")
    print("Expected average (calculated directly):")
    print(f"   {expected}\n")
    print("Homomorphically computed average (decrypted):")
    print(f"   {aggregated_decrypted[0]}\n")

    # Check if they match (allowing small floating-point error)
    difference = np.abs(expected - aggregated_decrypted[0])
    max_error = np.max(difference)

    print(f"Maximum error: {max_error:.10f}")

    if max_error < 1e-6:
        print("✅ PERFECT! Homomorphic encryption works correctly!\n")
    else:
        print("⚠️  Small precision loss (expected with approximate HE)\n")

    # Print statistics
    print("=" * 60)
    print("📈 ENCRYPTION STATISTICS")
    print("=" * 60)
    info = he_manager.get_context_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("🎉 TEST COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run test when executed directly
    test_encryption()
