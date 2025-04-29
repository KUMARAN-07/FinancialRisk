import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any
import joblib
from pathlib import Path
import logging
from datetime import datetime
from dateutil.parser import parse as parse_date

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
            # Handle timestamp conversion - could be string, datetime or Neo4j DateTime
            timestamp = tx['timestamp']
            if isinstance(timestamp, str):
                timestamp = parse_date(timestamp)
            elif hasattr(timestamp, 'to_native'):  # Neo4j DateTime
                timestamp = timestamp.to_native()
            
            # Extract time-based features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Transaction features
            amount = float(tx['amount'])
            
            # Category encoding (optional)
            category_risk = 0
            if 'category' in tx:
                high_risk_categories = ['GAMBLING', 'CRYPTO', 'LUXURY', 'TRANSFER']
                medium_risk_categories = ['TRAVEL', 'ENTERTAINMENT']
                if tx['category'] in high_risk_categories:
                    category_risk = 2
                elif tx['category'] in medium_risk_categories:
                    category_risk = 1
            
            # Combine features
            feature_vector = [
                hour,
                day_of_week,
                is_weekend,
                amount,
                category_risk
            ]
            features.append(feature_vector)
            
            # Log feature extraction for debugging
            logger.debug(f"Features for tx {tx.get('id', 'unknown')}: {feature_vector}")
        
        return np.array(features)

    def fit(self, transactions: List[Dict[str, Any]]) -> None:
        """Train the anomaly detection model."""
        try:
            X = self.preprocess_features(transactions)
            logger.info(f"Training anomaly model on {len(transactions)} transactions with feature shape {X.shape}")
            self.model.fit(X)
            self.is_fitted = True
            logger.info("Anomaly detection model trained successfully")
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {str(e)}")
            raise

    def predict(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict anomalies in transactions."""
        if not self.is_fitted:
            logger.warning("Model not fitted, returning transactions without anomaly detection")
            for tx in transactions:
                tx['is_anomaly'] = False
                tx['anomaly_score'] = 0.0
            return transactions
        
        try:
            X = self.preprocess_features(transactions)
            logger.info(f"Predicting anomalies for {len(transactions)} transactions with feature shape {X.shape}")
            
            # For isolation forest, lower scores (more negative) = more anomalous
            scores = self.model.score_samples(X)  
            predictions = self.model.predict(X)
            
            # Log distribution of scores
            logger.info(f"Anomaly score stats - min: {np.min(scores)}, max: {np.max(scores)}, mean: {np.mean(scores)}")
            
            # Convert predictions to results
            results = []
            for i, (tx, pred, score) in enumerate(zip(transactions, predictions, scores)):
                result = tx.copy()
                # In isolation forest: -1 = anomaly, 1 = normal
                result['is_anomaly'] = pred == -1  
                result['anomaly_score'] = float(score)
                
                # Force anomaly detection for very high amounts as a fallback
                if 'amount' in tx and float(tx['amount']) > 5000:
                    high_risk_categories = ['GAMBLING', 'CRYPTO', 'LUXURY', 'TRANSFER']
                    if 'category' in tx and tx['category'] in high_risk_categories:
                        logger.info(f"Forcing anomaly detection for high-risk transaction {tx.get('id', 'unknown')}")
                        result['is_anomaly'] = True
                        # Make score more negative to indicate anomaly
                        if score > -0.2:  
                            result['anomaly_score'] = -0.5
                
                logger.info(f"Transaction {i+1}/{len(transactions)} (ID: {tx.get('id', 'unknown')}): "
                           f"amount={tx.get('amount', 'N/A')}, category={tx.get('category', 'N/A')}, "
                           f"is_anomaly={result['is_anomaly']}, score={result['anomaly_score']}")
                results.append(result)
            
            # Log overall results
            anomaly_count = sum(1 for r in results if r['is_anomaly'])
            logger.info(f"Detected {anomaly_count} anomalies out of {len(results)} transactions")
            
            return results
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            # Return transactions without anomaly detection in case of error
            for tx in transactions:
                tx['is_anomaly'] = False
                tx['anomaly_score'] = 0.0
            return transactions

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