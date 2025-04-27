import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42, n_estimators: int = 100):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators
        )
        self.is_fitted = False

    def preprocess_features(self, transactions: List[Dict[str, Any]]) -> np.ndarray:
        """Convert transaction data into feature matrix."""
        features = []
        for tx in transactions:
            # Extract time-based features
            hour = tx['timestamp'].hour
            day_of_week = tx['timestamp'].weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Transaction features
            amount = tx['amount']
            
            # Combine features
            feature_vector = [
                hour,
                day_of_week,
                is_weekend,
                amount
            ]
            features.append(feature_vector)
        
        return np.array(features)

    def fit(self, transactions: List[Dict[str, Any]]) -> None:
        """Train the anomaly detection model."""
        try:
            X = self.preprocess_features(transactions)
            self.model.fit(X)
            self.is_fitted = True
            logger.info("Anomaly detection model trained successfully")
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {str(e)}")
            raise

    def predict(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict anomalies in transactions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            X = self.preprocess_features(transactions)
            predictions = self.model.predict(X)
            scores = self.model.score_samples(X)
            
            # Convert predictions to results
            results = []
            for tx, pred, score in zip(transactions, predictions, scores):
                result = tx.copy()
                result['is_anomaly'] = pred == -1
                result['anomaly_score'] = float(score)
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("No trained model to save")
        
        try:
            model_path = Path(path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        try:
            self.model = joblib.load(path)
            self.is_fitted = True
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 