import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from financial_risk.models.anomaly.isolation_forest import AnomalyDetector
from financial_risk.models.clustering.kmeans import BehavioralClusterer
from financial_risk.graph.models import GraphDatabase
from financial_risk.utils.data_generator import generate_test_data

def test_anomaly_detection():
    """Test the anomaly detection functionality."""
    # Generate test data
    transactions = generate_test_data(
        n_customers=10,
        n_merchants=5,
        n_transactions=100,
        anomaly_ratio=0.1
    )
    
    # Initialize and train anomaly detector
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(transactions)
    
    # Make predictions
    results = detector.predict(transactions)
    
    # Verify results
    assert len(results) == len(transactions)
    assert all('is_anomaly' in result for result in results)
    assert all('anomaly_score' in result for result in results)
    
    # Check if anomalies were detected
    n_anomalies = sum(1 for result in results if result['is_anomaly'])
    assert n_anomalies > 0, "No anomalies detected in test data"

def test_behavioral_clustering():
    """Test the behavioral clustering functionality."""
    # Generate test data
    transactions = generate_test_data(
        n_customers=10,
        n_merchants=5,
        n_transactions=100,
        anomaly_ratio=0.1
    )
    
    # Get unique customer IDs
    customer_ids = list(set(tx['customer_id'] for tx in transactions))
    
    # Initialize and train clusterer
    clusterer = BehavioralClusterer(n_clusters=3)
    clusterer.fit(transactions, customer_ids)
    
    # Test predictions for each customer
    for customer_id in customer_ids:
        cluster, distance = clusterer.predict(transactions, customer_id)
        assert isinstance(cluster, int)
        assert isinstance(distance, float)
        assert 0 <= cluster < 3
        assert distance >= 0

def test_graph_database():
    """Test the graph database functionality."""
    # Initialize graph database
    graph_db = GraphDatabase(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password",
        database="financial_risk_test"
    )
    
    # Generate test data
    transactions = generate_test_data(
        n_customers=5,
        n_merchants=3,
        n_transactions=20,
        anomaly_ratio=0.1
    )
    
    # Test node creation
    for tx in transactions[:5]:  # Test with first 5 transactions
        # Create nodes
        customer = graph_db.create_customer({
            'id': tx['customer_id'],
            'name': f"Test_Customer_{tx['customer_id']}",
            'email': f"test_{tx['customer_id']}@example.com"
        })
        
        account = graph_db.create_account({
            'id': tx['account_id'],
            'customer_id': tx['customer_id'],
            'account_type': 'SAVINGS',
            'balance': 0.0
        })
        
        merchant = graph_db.create_merchant({
            'id': tx['merchant_id'],
            'name': f"Test_Merchant_{tx['merchant_id']}",
            'category': tx['category']
        })
        
        tx_node = graph_db.create_transaction(tx)
        
        # Create relationships
        graph_db.create_relationship(customer, account, "HAS_ACCOUNT")
        graph_db.create_relationship(account, tx_node, "MADE_TRANSACTION")
        graph_db.create_relationship(tx_node, merchant, "TO")
    
    # Test queries
    customer_id = transactions[0]['customer_id']
    customer_txs = graph_db.get_customer_transactions(customer_id)
    assert len(customer_txs) > 0
    
    merchant_id = transactions[0]['merchant_id']
    merchant_txs = graph_db.get_merchant_transactions(merchant_id)
    assert len(merchant_txs) > 0
    
    patterns = graph_db.get_customer_behavioral_patterns(customer_id)
    assert 'transaction_count' in patterns
    assert 'avg_amount' in patterns
    
    risk_score = graph_db.get_merchant_risk_score(merchant_id)
    assert 0 <= risk_score <= 1

if __name__ == "__main__":
    pytest.main([__file__]) 