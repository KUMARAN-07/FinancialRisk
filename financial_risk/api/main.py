from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from financial_risk.utils.data_generator import generate_test_data
from financial_risk.graph.models import get_or_create_customer, get_or_create_account, get_or_create_merchant
from typing import List, Dict, Any
import logging
from datetime import datetime
import yaml
from pathlib import Path
from dateutil.parser import isoparse

from ..models.base import Transaction, Customer, Account, Merchant
from ..models.anomaly.isolation_forest import AnomalyDetector
from ..models.clustering.kmeans import BehavioralClusterer
from ..graph.models import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title=config['api']['title'],
    version=config['api']['version']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
graph_db = GraphDatabase(
    uri=config['neo4j']['uri'],
    user=config['neo4j']['user'],
    password=config['neo4j']['password'],
    database=config['neo4j']['database']
)

anomaly_detector = AnomalyDetector(
    contamination=config['models']['anomaly']['isolation_forest']['contamination'],
    random_state=config['models']['anomaly']['isolation_forest']['random_state'],
    n_estimators=config['models']['anomaly']['isolation_forest']['n_estimators']
)

behavioral_clusterer = BehavioralClusterer(
    n_clusters=config['models']['kmeans']['n_clusters'],
    random_state=config['models']['kmeans']['random_state'],
    max_iter=config['models']['kmeans']['max_iter']
)

@app.post("/dev/generate_and_train")
async def generate_and_train(
    n_customers: int = 10,
    n_merchants: int = 5,
    n_transactions: int = 100,
    anomaly_ratio: float = 0.1
):
    """
    Generate synthetic transactions, insert into Neo4j, and train the models.
    """
    try:
        # 1. Generate synthetic data
        transactions = generate_test_data(
            n_customers=n_customers,
            n_merchants=n_merchants,
            n_transactions=n_transactions,
            anomaly_ratio=anomaly_ratio
        )
        # Convert datetime to ISO format for JSON serialization
        for tx in transactions:
            if hasattr(tx['timestamp'], 'isoformat'):
                tx['timestamp'] = tx['timestamp'].isoformat()

        # 2. Insert into Neo4j
        for tx in transactions:
            # Ensure all UUIDs are strings
            for key in ['id', 'customer_id', 'account_id', 'merchant_id']:
                if key in tx and not isinstance(tx[key], str):
                    tx[key] = str(tx[key])
            # Ensure all booleans are Python bool
            for key in ['is_anomaly']:
                if key in tx and type(tx[key]).__module__ == 'numpy':
                    tx[key] = bool(tx[key])
            tx_node = graph_db.create_transaction(tx)
            customer = get_or_create_customer(graph_db.graph, {
                'id': tx['customer_id'],
                'name': f"Customer_{tx['customer_id']}",
                'email': f"customer_{tx['customer_id']}@example.com"
            })
            account = get_or_create_account(graph_db.graph, {
                'id': tx['account_id'],
                'customer_id': tx['customer_id'],
                'account_type': 'SAVINGS',
                'balance': 0.0
            })
            merchant = get_or_create_merchant(graph_db.graph, {
                'id': tx['merchant_id'],
                'name': f"Merchant_{tx['merchant_id']}",
                'category': tx['category']
            })
            graph_db.create_relationship(customer, account, "HAS_ACCOUNT")
            graph_db.create_relationship(account, tx_node, "MADE_TRANSACTION")
            graph_db.create_relationship(tx_node, merchant, "TO")

        # 3. Train the models
        query = """
        MATCH (t:Transaction)
        RETURN t
        """
        result = graph_db.graph.run(query)
        all_transactions = [dict(record["t"]) for record in result]

        # Convert timestamp strings to datetime objects
        for tx in all_transactions:
            # If it's a Neo4j DateTime, convert to Python datetime
            if hasattr(tx['timestamp'], 'to_native'):
                tx['timestamp'] = tx['timestamp'].to_native()
            # If it's a string, parse to datetime
            elif isinstance(tx['timestamp'], str):
                tx['timestamp'] = isoparse(tx['timestamp'])

        anomaly_detector.fit(all_transactions)
        customer_ids = list(set(tx['customer_id'] for tx in all_transactions))
        behavioral_clusterer.fit(all_transactions, customer_ids)

        return {
            "status": "success",
            "inserted_transactions": len(transactions),
            "trained_on_transactions": len(all_transactions)
        }
    except Exception as e:
        logger.error(f"Error in generate_and_train: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/transactions/process", response_model=List[Dict[str, Any]])
async def process_transactions(transactions: List[Transaction]):
    """Process a batch of transactions and detect anomalies."""
    try:
        # Convert transactions to dict for processing
        tx_dicts = [tx.dict() for tx in transactions]
        
        # Detect anomalies
        results = anomaly_detector.predict(tx_dicts)
        
        # Store in graph database
        for tx in results:
            # Ensure all UUIDs are strings
            for key in ['id', 'customer_id', 'account_id', 'merchant_id']:
                if key in tx and not isinstance(tx[key], str):
                    tx[key] = str(tx[key])
            # Ensure all booleans are Python bool
            for key in ['is_anomaly']:
                if key in tx and type(tx[key]).__module__ == 'numpy':
                    tx[key] = bool(tx[key])
            # Create transaction node
            tx_node = graph_db.create_transaction(tx)
            
            # Create or get customer node
            customer = get_or_create_customer(graph_db.graph, {
                'id': tx['customer_id'],
                'name': f"Customer_{tx['customer_id']}",
                'email': f"customer_{tx['customer_id']}@example.com"
            })
            
            # Create or get account node
            account = get_or_create_account(graph_db.graph, {
                'id': tx['account_id'],
                'customer_id': tx['customer_id'],
                'account_type': 'SAVINGS',
                'balance': 0.0
            })
            
            # Create or get merchant node
            merchant = get_or_create_merchant(graph_db.graph, {
                'id': tx['merchant_id'],
                'name': f"Merchant_{tx['merchant_id']}",
                'category': tx['category']
            })
            
            # Create relationships
            graph_db.create_relationship(customer, account, "HAS_ACCOUNT")
            graph_db.create_relationship(account, tx_node, "MADE_TRANSACTION")
            graph_db.create_relationship(tx_node, merchant, "TO")
        
        return results
    except Exception as e:
        logger.error(f"Error processing transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customers/{customer_id}/behavior")
async def get_customer_behavior(customer_id: str):
    """Get behavioral patterns and cluster for a customer."""
    try:
        # Get customer transactions
        transactions = graph_db.get_customer_transactions(customer_id)
        
        # Convert timestamp strings to datetime objects
        for tx in transactions:
            if isinstance(tx['timestamp'], str):
                tx['timestamp'] = isoparse(tx['timestamp'])
            elif hasattr(tx['timestamp'], 'to_native'):
                tx['timestamp'] = tx['timestamp'].to_native()
        
        # Get behavioral patterns
        patterns = graph_db.get_customer_behavioral_patterns(customer_id)
        
        # Get cluster prediction
        cluster, distance = behavioral_clusterer.predict(transactions, customer_id)
        
        # Convert NumPy types to Python types
        if hasattr(cluster, 'item'):
            cluster = cluster.item()
        if hasattr(distance, 'item'):
            distance = distance.item()
        
        return {
            "customer_id": customer_id,
            "behavioral_patterns": patterns,
            "cluster": cluster,
            "cluster_distance": distance
        }
    except Exception as e:
        logger.error(f"Error getting customer behavior: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/merchants/{merchant_id}/risk")
async def get_merchant_risk(merchant_id: str):
    """Get risk score and transaction patterns for a merchant."""
    try:
        # Get merchant transactions
        transactions = graph_db.get_merchant_transactions(merchant_id)
        
        # Calculate risk score
        risk_score = graph_db.get_merchant_risk_score(merchant_id)
        
        return {
            "merchant_id": merchant_id,
            "risk_score": risk_score,
            "transaction_count": len(transactions)
        }
    except Exception as e:
        logger.error(f"Error getting merchant risk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/train")
async def train_models():
    """Train the anomaly detection and clustering models."""
    try:
        # Get all transactions from graph database
        query = """
        MATCH (t:Transaction)
        RETURN t
        """
        result = graph_db.graph.run(query)
        transactions = [dict(record["t"]) for record in result]
        
        # Convert timestamp strings to datetime objects
        for tx in transactions:
            # If it's a Neo4j DateTime, convert to Python datetime
            if hasattr(tx['timestamp'], 'to_native'):
                tx['timestamp'] = tx['timestamp'].to_native()
            # If it's a string, parse to datetime
            elif isinstance(tx['timestamp'], str):
                tx['timestamp'] = isoparse(tx['timestamp'])

        # Train anomaly detector
        anomaly_detector.fit(transactions)
        
        # Get unique customer IDs
        customer_ids = list(set(tx['customer_id'] for tx in transactions))
        
        # Train behavioral clusterer
        behavioral_clusterer.fit(transactions, customer_ids)
        
        return {"status": "success", "message": "Models trained successfully"}
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['debug']
    ) 