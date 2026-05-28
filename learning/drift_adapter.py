"""Drift Detection & Adaptation for changing task distributions.

This module implements GMM-based density estimation with Page-Hinkley statistical test
for detecting concept drift in task patterns.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass


class PageHinkleyTest:
    """Page-Hinkley statistical test for change detection."""
    
    def __init__(self, delta: float = 0.05, threshold: float = 50):
        self.delta = delta  # Minimum magnitude of allowed change
        self.threshold = threshold  # Detection threshold
        
        self.m_n = 0.0  # Cumulative sum
        self.x_mean = 0.0  # Running mean
        self.n = 0  # Sample count
        self.drift_detected = False
        self.min_value = float('inf')
    
    def update(self, value: float) -> bool:
        """Update test with new observation and check for drift."""
        self.n += 1
        
        # Update running mean
        self.x_mean += (value - self.x_mean) / self.n
        
        # Update cumulative sum
        self.m_n += value - self.x_mean - self.delta
        
        # Track minimum value
        self.min_value = min(self.min_value, self.m_n)
        
        # Check for drift
        ph_test = self.m_n - self.min_value
        self.drift_detected = ph_test > self.threshold
        
        return self.drift_detected
    
    def reset(self):
        """Reset test after drift detection."""
        self.m_n = 0.0
        self.x_mean = 0.0
        self.n = 0
        self.drift_detected = False
        self.min_value = float('inf')


class DriftDetector:
    """GMM + statistical test for concept drift."""
    
    def __init__(self, feature_dim: int = 384, window_size: int = 100):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        
        # GMM for density estimation
        try:
            from sklearn.mixture import GaussianMixture
            self.gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            self.gmm = None
        
        # Page-Hinkley test for success rate monitoring
        self.page_hinkley = PageHinkleyTest(delta=0.05, threshold=50)
        
        self.last_density: Optional[float] = None
        self.fitted = False
    
    def update(self, feature_vector: np.ndarray, success: bool):
        """Add observation and check for drift."""
        self.window.append({"features": feature_vector, "success": success})
        
        # Update density model periodically when window is full
        if len(self.window) == self.window_size and self.sklearn_available:
            X = np.array([w["features"] for w in self.window])
            try:
                self.gmm.fit(X)
                self.last_density = self.gmm.score_samples([feature_vector])[0]
                self.fitted = True
            except Exception:
                pass
        
        # Update statistical test on success rate
        self.page_hinkley.update(1.0 if success else 0.0)
    
    def is_drift_detected(self, new_features: np.ndarray) -> bool:
        """Check if new observation is outlier or test signals drift."""
        # Check Page-Hinkley test first
        if self.page_hinkley.drift_detected:
            return True
        
        # Density-based drift detection
        if self.fitted and self.last_density is not None and self.sklearn_available:
            try:
                new_density = self.gmm.score_samples([new_features])[0]
                
                # Significant drop in density indicates drift
                if new_density < self.last_density - 2.0:  # 2 std dev threshold
                    return True
            except Exception:
                pass
        
        return False
    
    def get_adaptation_action(self) -> str:
        """Return recommended action when drift detected."""
        if self.page_hinkley.drift_detected:
            return "retrain"  # Success rate has changed significantly
        
        return "deprecate_workflows"  # Distribution has shifted
    
    def get_drift_severity(self) -> float:
        """Estimate severity of detected drift."""
        if not self.page_hinkley.drift_detected:
            return 0.0
        
        # Severity based on how much PH test exceeds threshold
        ph_value = self.page_hinkley.m_n - self.page_hinkley.min_value
        severity = min(1.0, (ph_value - self.page_hinkley.threshold) / self.page_hinkley.threshold)
        
        return severity
    
    def reset_after_adaptation(self):
        """Reset detector after adaptation action is taken."""
        self.page_hinkley.reset()
        self.last_density = None
        self.fitted = False
        # Keep window but clear GMM fit state


class DriftAdapter:
    """Manage adaptation strategy when drift is detected."""
    
    def __init__(self, config=None):
        self.config = config
        self.detector: Optional[DriftDetector] = None
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def initialize_detector(self, feature_dim: int = 384, window_size: int = 100):
        """Initialize drift detector with specified parameters."""
        self.detector = DriftDetector(feature_dim, window_size)
    
    def record_observation(self, features: np.ndarray, success: bool):
        """Record observation and check for drift."""
        if self.detector:
            self.detector.update(features, success)
    
    def check_drift(self, new_features: np.ndarray) -> tuple:
        """Check for drift and return (detected, action)."""
        if not self.detector:
            return False, "none"
        
        detected = self.detector.is_drift_detected(new_features)
        action = self.detector.get_adaptation_action() if detected else "none"
        
        if detected:
            self.adaptation_history.append({
                "timestamp": __import__('time').time(),
                "action": action,
                "severity": self.detector.get_drift_severity(),
            })
        
        return detected, action
    
    def recommend_adaptation_strategy(self) -> Dict[str, Any]:
        """Recommend comprehensive adaptation strategy."""
        if not self.detector:
            return {"action": "none", "details": "No detector initialized"}
        
        action = self.detector.get_adaptation_action()
        
        strategies = {
            "deprecate_workflows": {
                "action": "deprecate",
                "details": "Deprecate old workflows and collect new examples",
                "collect_new_samples": True,
                "fallback_to_llm": True,
            },
            "retrain": {
                "action": "retrain", 
                "details": "Retrain models on recent data",
                "use_recent_window": True,
                "min_samples": 50,
            },
            "none": {
                "action": "continue",
                "details": "No adaptation needed",
            }
        }
        
        return strategies.get(action, strategies["none"])
