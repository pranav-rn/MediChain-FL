"""
Utility modules for MediChain-FL
- Homomorphic encryption (TenSEAL CKKS)
- Anomaly detection
"""

from .encryption import HEManager
from .anomaly_detector import GradientAnomalyDetector

__all__ = [
    'HEManager',
    'GradientAnomalyDetector',
]