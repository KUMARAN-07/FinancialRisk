import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple
import joblib
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BehavioralClusterer:
    def __init__(self, n_clusters: int = 5, random_state: int = 42, max_iter: int = 300):
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter
        )
        self.is_fitted = False
        self.cluster_centers_ = None
        self.cluster_labels_ = None

    def calculate_customer_features(self, transactions: List[Dict[str, Any]], 
                                  customer_id: str) -> np.ndarray:
        """Calculate behavioral features for a customer."""
        customer_txs = [tx for tx in transactions if tx['customer_id'] == customer_id]
        
        if not customer_txs:
            return np.zeros(5)  # Return zero vector if no transactions
        
        # Calculate time-based features
        tx_times = [tx['timestamp'] for tx in customer_txs]
        avg_hour = np.mean([t.hour for t in tx_times])
        weekend_ratio = sum(1 for t in tx_times if t.weekday() >= 5) / len(tx_times)
        
        # Calculate transaction features
        amounts = [tx['amount'] for tx in customer_txs]
        avg_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        # Calculate frequency
        if len(tx_times) > 1:
            time_diff = max(tx_times) - min(tx_times)
            frequency = len(tx_times) / (time_diff.days + 1)  # Transactions per day
        else:
            frequency = 0
        
        return np.array([
            avg_hour,
            weekend_ratio,
            avg_amount,
            std_amount,
            frequency
        ])

    def fit(self, transactions: List[Dict[str, Any]], customer_ids: List[str]) -> None:
        """Train the clustering model."""
        try:
            # Calculate features for each customer
            features = []
            for customer_id in customer_ids:
                customer_features = self.calculate_customer_features(transactions, customer_id)
                features.append(customer_features)
            
            X = np.array(features)
            
            # Fit the model
            self.model.fit(X)
            self.cluster_centers_ = self.model.cluster_centers_
            self.cluster_labels_ = self.model.labels_
            self.is_fitted = True
            
            logger.info("Behavioral clustering model trained successfully")
        except Exception as e:
            logger.error(f"Error training clustering model: {str(e)}")
            raise

    def predict(self, transactions: List[Dict[str, Any]], customer_id: str) -> Tuple[int, float]:
        """Predict the behavioral cluster for a customer."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Calculate features for the customer
            features = self.calculate_customer_features(transactions, customer_id)
            
            # Predict cluster
            cluster = self.model.predict([features])[0]
            
            # Calculate distance to cluster center
            distance = np.linalg.norm(features - self.cluster_centers_[cluster])
            
            return cluster, float(distance)
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def get_cluster_characteristics(self) -> List[Dict[str, Any]]:
        """Get characteristics of each cluster."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster characteristics")
        
        characteristics = []
        for i, center in enumerate(self.cluster_centers_):
            char = {
                'cluster_id': i,
                'avg_hour': center[0],
                'weekend_ratio': center[1],
                'avg_amount': center[2],
                'amount_std': center[3],
                'frequency': center[4]
            }
            characteristics.append(char)
        
        return characteristics

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("No trained model to save")
        
        try:
            model_path = Path(path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                'model': self.model,
                'cluster_centers': self.cluster_centers_,
                'cluster_labels': self.cluster_labels_
            }, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        try:
            saved_data = joblib.load(path)
            self.model = saved_data['model']
            self.cluster_centers_ = saved_data['cluster_centers']
            self.cluster_labels_ = saved_data['cluster_labels']
            self.is_fitted = True
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 