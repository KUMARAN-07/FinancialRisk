import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import uuid4
import numpy as np

class TransactionGenerator:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.merchant_categories = [
            "GROCERY", "RESTAURANT", "RETAIL", "UTILITIES", 
            "ENTERTAINMENT", "TRAVEL", "HEALTHCARE", "EDUCATION"
        ]
        
        # Define normal spending patterns
        self.category_means = {
            "GROCERY": 1000,
            "RESTAURANT": 500,
            "RETAIL": 2000,
            "UTILITIES": 3000,
            "ENTERTAINMENT": 800,
            "TRAVEL": 5000,
            "HEALTHCARE": 1500,
            "EDUCATION": 10000
        }
        
        self.category_stds = {
            "GROCERY": 200,
            "RESTAURANT": 100,
            "RETAIL": 500,
            "UTILITIES": 300,
            "ENTERTAINMENT": 200,
            "TRAVEL": 1000,
            "HEALTHCARE": 300,
            "EDUCATION": 2000
        }

    def generate_customer_ids(self, n_customers: int) -> List[str]:
        """Generate unique customer IDs."""
        return [str(uuid4()) for _ in range(n_customers)]

    def generate_account_ids(self, customer_ids: List[str]) -> Dict[str, List[str]]:
        """Generate account IDs for each customer."""
        return {
            customer_id: [str(uuid4()) for _ in range(random.randint(1, 3))]
            for customer_id in customer_ids
        }

    def generate_merchant_ids(self, n_merchants: int) -> List[str]:
        """Generate unique merchant IDs."""
        return [str(uuid4()) for _ in range(n_merchants)]

    def generate_transaction_time(self) -> datetime:
        """Generate a random transaction time within the date range."""
        time_diff = self.end_date - self.start_date
        random_seconds = random.randint(0, int(time_diff.total_seconds()))
        return self.start_date + timedelta(seconds=random_seconds)

    def generate_normal_transaction(self, customer_id: str, account_id: str, 
                                  merchant_id: str, category: str) -> Dict[str, Any]:
        """Generate a normal transaction."""
        amount = np.random.normal(
            self.category_means[category],
            self.category_stds[category]
        )
        amount = max(0, amount)  # Ensure non-negative amount
        
        return {
            "id": str(uuid4()),
            "customer_id": customer_id,
            "account_id": account_id,
            "merchant_id": merchant_id,
            "amount": round(amount, 2),
            "timestamp": self.generate_transaction_time(),
            "category": category,
            "description": f"Transaction at {category.lower()} merchant",
            "is_anomaly": False,
            "anomaly_score": None
        }

    def generate_anomalous_transaction(self, customer_id: str, account_id: str,
                                     merchant_id: str, category: str) -> Dict[str, Any]:
        """Generate an anomalous transaction."""
        # Generate amount with higher variance
        amount = np.random.normal(
            self.category_means[category] * 3,  # Higher mean
            self.category_stds[category] * 2    # Higher variance
        )
        amount = max(0, amount)  # Ensure non-negative amount
        
        return {
            "id": str(uuid4()),
            "customer_id": customer_id,
            "account_id": account_id,
            "merchant_id": merchant_id,
            "amount": round(amount, 2),
            "timestamp": self.generate_transaction_time(),
            "category": category,
            "description": f"Anomalous transaction at {category.lower()} merchant",
            "is_anomaly": True,
            "anomaly_score": None
        }

    def generate_transactions(self, n_customers: int = 100, n_merchants: int = 50,
                            n_transactions: int = 1000, anomaly_ratio: float = 0.1) -> List[Dict[str, Any]]:
        """Generate a set of transactions with some anomalies."""
        # Generate IDs
        customer_ids = self.generate_customer_ids(n_customers)
        account_ids = self.generate_account_ids(customer_ids)
        merchant_ids = self.generate_merchant_ids(n_merchants)
        
        # Generate transactions
        transactions = []
        n_anomalies = int(n_transactions * anomaly_ratio)
        
        for _ in range(n_transactions):
            # Select random IDs
            customer_id = random.choice(customer_ids)
            account_id = random.choice(account_ids[customer_id])
            merchant_id = random.choice(merchant_ids)
            category = random.choice(self.merchant_categories)
            
            # Generate transaction
            if len(transactions) < n_anomalies:
                transaction = self.generate_anomalous_transaction(
                    customer_id, account_id, merchant_id, category
                )
            else:
                transaction = self.generate_normal_transaction(
                    customer_id, account_id, merchant_id, category
                )
            
            transactions.append(transaction)
        
        # Sort by timestamp
        transactions.sort(key=lambda x: x['timestamp'])
        return transactions

def generate_test_data(n_customers: int = 100, n_merchants: int = 50,
                      n_transactions: int = 1000, anomaly_ratio: float = 0.1,
                      days: int = 30) -> List[Dict[str, Any]]:
    """Generate test data for the specified parameters."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    generator = TransactionGenerator(start_date, end_date)
    return generator.generate_transactions(
        n_customers=n_customers,
        n_merchants=n_merchants,
        n_transactions=n_transactions,
        anomaly_ratio=anomaly_ratio
    ) 