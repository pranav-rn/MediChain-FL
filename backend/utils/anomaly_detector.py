"""
Gradient Anomaly Detection
Detects suspicious model updates that may indicate:
- Model poisoning attacks
- Byzantine behavior
- Data quality issues
- Adversarial manipulation

Method: Statistical outlier detection using Z-score
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Union
import torch


class GradientAnomalyDetector:
    """
    Statistical anomaly detector for federated learning gradients
    
    Uses Z-score method to detect outliers:
    - Tracks gradient L2 norms over time
    - Calculates mean and standard deviation
    - Flags updates with Z-score > threshold
    """
    
    def __init__(self, threshold: float = 3.0, history_size: int = 20):
        """
        Initialize anomaly detector
        
        Args:
            threshold: Z-score threshold (typically 2.0-3.0)
                      3.0 = 99.7% confidence interval
            history_size: Number of past updates to track
        """
        self.threshold = threshold
        self.history = deque(maxlen=history_size)
        self.detected_anomalies = []
        
        print(f"🚨 Anomaly Detector initialized")
        print(f"   Z-score threshold: {threshold}")
        print(f"   History size: {history_size}")
        print()
    
    def check_update(
        self,
        gradients: Union[List[np.ndarray], List[torch.Tensor]],
        hospital_id: str,
        round_num: int
    ) -> Tuple[bool, float, str]:
        """
        Check if gradient update is anomalous
        
        Args:
            gradients: List of gradient tensors
            hospital_id: Hospital identifier
            round_num: Current training round
        
        Returns:
            Tuple of (is_anomalous, z_score, reason)
        """
        
        # Calculate L2 norm of gradients
        gradient_norm = self._calculate_gradient_norm(gradients)
        
        # Need at least 3 samples for meaningful statistics
        if len(self.history) < 3:
            self.history.append(gradient_norm)
            return False, 0.0, "Insufficient history (building baseline)"
        
        # Calculate Z-score
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        # Avoid division by zero
        if std < 1e-8:
            z_score = 0.0
        else:
            z_score = abs((gradient_norm - mean) / std)
        
        # Check if anomalous
        is_anomalous = z_score > self.threshold
        
        # Add to history
        self.history.append(gradient_norm)
        
        # Log if anomalous
        if is_anomalous:
            reason = f"Z-score {z_score:.2f} exceeds threshold {self.threshold}"
            
            self.detected_anomalies.append({
                'hospital_id': hospital_id,
                'round': round_num,
                'z_score': z_score,
                'gradient_norm': gradient_norm,
                'mean': mean,
                'std': std,
                'reason': reason
            })
            
            print(f"⚠️  ANOMALY DETECTED!")
            print(f"   Hospital: {hospital_id}")
            print(f"   Round: {round_num}")
            print(f"   Z-score: {z_score:.2f} (threshold: {self.threshold})")
            print(f"   Gradient norm: {gradient_norm:.4f}")
            print(f"   Expected range: [{mean - 3*std:.4f}, {mean + 3*std:.4f}]")
            print(f"   Reason: {reason}\n")
            
            return True, z_score, reason
        
        return False, z_score, "Normal"
    
    def _calculate_gradient_norm(
        self,
        gradients: Union[List[np.ndarray], List[torch.Tensor]]
    ) -> float:
        """
        Calculate L2 norm of all gradients
        
        Args:
            gradients: List of gradient tensors
        
        Returns:
            L2 norm (scalar)
        """
        total_norm = 0.0
        
        for grad in gradients:
            # Convert to numpy if PyTorch tensor
            if isinstance(grad, torch.Tensor):
                grad = grad.cpu().detach().numpy()
            
            # Calculate L2 norm and add to total
            norm = np.linalg.norm(grad)
            total_norm += norm ** 2
        
        return np.sqrt(total_norm)
    
    def get_statistics(self) -> Dict:
        """
        Get detector statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_checked': len(self.history),
            'anomalies_detected': len(self.detected_anomalies),
            'current_mean': float(np.mean(self.history)) if len(self.history) > 0 else 0.0,
            'current_std': float(np.std(self.history)) if len(self.history) > 0 else 0.0,
            'threshold': self.threshold,
            'history_size': self.history.maxlen
        }
    
    def get_anomaly_report(self) -> List[Dict]:
        """
        Get list of all detected anomalies
        
        Returns:
            List of anomaly dictionaries
        """
        return self.detected_anomalies
    
    def reset(self):
        """Reset detector state"""
        self.history.clear()
        self.detected_anomalies.clear()
        print("🔄 Anomaly detector reset\n")


def test_anomaly_detector():
    """Test anomaly detector with simulated data"""
    
    print("\n" + "="*60)
    print("🧪 TESTING ANOMALY DETECTOR")
    print("="*60 + "\n")
    
    detector = GradientAnomalyDetector(threshold=3.0)
    
    # Simulate 10 normal updates
    print("📊 Simulating 10 NORMAL updates...\n")
    for i in range(10):
        # Normal gradients (small values around 0)
        normal_grads = [np.random.randn(100) * 0.01]
        is_anomalous, z_score, reason = detector.check_update(
            normal_grads, f"Hospital_{i%3 + 1}", i
        )
        
        if not is_anomalous:
            print(f"   Round {i}: Normal (Z-score: {z_score:.2f})")
    
    print("\n⚠️  Simulating 1 POISONED update...\n")
    
    # Poisoned update (large gradients)
    poisoned_grads = [np.random.randn(100) * 10.0]  # 100x larger!
    is_anomalous, z_score, reason = detector.check_update(
        poisoned_grads, "Hospital_Malicious", 11
    )
    
    if is_anomalous:
        print("✅ Anomaly successfully detected!\n")
    else:
        print("❌ Failed to detect anomaly\n")
    
    # Print statistics
    stats = detector.get_statistics()
    print("="*60)
    print("📈 DETECTOR STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("🎉 TEST COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_anomaly_detector()