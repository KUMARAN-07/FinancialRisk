from py2neo import Graph, Node, Relationship, NodeMatcher
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)

class GraphDatabase:
    def __init__(self, uri: str, user: str, password: str, database: str = "financial_risk"):
        self.graph = Graph(uri, auth=(user, password), name=database)
        self._create_constraints()

    def _create_constraints(self):
        """Create unique constraints for nodes."""
        try:
            # Create constraints if they don't exist
            self.graph.run("CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE")
            self.graph.run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
            self.graph.run("CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE")
            self.graph.run("CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.id IS UNIQUE")
        except Exception as e:
            logger.error(f"Error creating constraints: {str(e)}")
            raise

    def create_customer(self, customer_data: Dict[str, Any]) -> Node:
        """Create a customer node."""
        try:
            customer = Node("Customer", **customer_data)
            self.graph.create(customer)
            return customer
        except Exception as e:
            logger.error(f"Error creating customer: {str(e)}")
            raise

    def create_account(self, account_data: Dict[str, Any]) -> Node:
        """Create an account node."""
        try:
            account = Node("Account", **account_data)
            self.graph.create(account)
            return account
        except Exception as e:
            logger.error(f"Error creating account: {str(e)}")
            raise

    def create_transaction(self, transaction_data: Dict[str, Any]) -> Node:
        """Create a transaction node."""
        try:
            transaction = Node("Transaction", **transaction_data)
            self.graph.create(transaction)
            return transaction
        except Exception as e:
            logger.error(f"Error creating transaction: {str(e)}")
            raise

    def create_merchant(self, merchant_data: Dict[str, Any]) -> Node:
        """Create a merchant node."""
        try:
            merchant = Node("Merchant", **merchant_data)
            self.graph.create(merchant)
            return merchant
        except Exception as e:
            logger.error(f"Error creating merchant: {str(e)}")
            raise

    def create_relationship(self, from_node: Node, to_node: Node, relationship_type: str, 
                          properties: Optional[Dict[str, Any]] = None) -> Relationship:
        """Create a relationship between two nodes."""
        try:
            if properties is None:
                properties = {}
            relationship = Relationship(from_node, relationship_type, to_node, **properties)
            self.graph.create(relationship)
            return relationship
        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            raise

    def get_customer_transactions(self, customer_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all transactions for a customer."""
        try:
            query = """
            MATCH (c:Customer {id: $customer_id})-[:HAS_ACCOUNT]->(a:Account)-[:MADE_TRANSACTION]->(t:Transaction)
            RETURN t
            ORDER BY t.timestamp DESC
            LIMIT $limit
            """
            result = self.graph.run(query, customer_id=customer_id, limit=limit)
            return [dict(record["t"]) for record in result]
        except Exception as e:
            logger.error(f"Error getting customer transactions: {str(e)}")
            raise

    def get_merchant_transactions(self, merchant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all transactions for a merchant."""
        try:
            query = """
            MATCH (m:Merchant {id: $merchant_id})<-[:TO]-(t:Transaction)
            RETURN t
            ORDER BY t.timestamp DESC
            LIMIT $limit
            """
            result = self.graph.run(query, merchant_id=merchant_id, limit=limit)
            return [dict(record["t"]) for record in result]
        except Exception as e:
            logger.error(f"Error getting merchant transactions: {str(e)}")
            raise

    def get_customer_behavioral_patterns(self, customer_id: str) -> Dict[str, Any]:
        """Get behavioral patterns for a customer."""
        try:
            query = """
            MATCH (c:Customer {id: $customer_id})-[:HAS_ACCOUNT]->(a:Account)-[:MADE_TRANSACTION]->(t:Transaction)
            WITH c, a, t
            RETURN 
                c.id as customer_id,
                count(t) as transaction_count,
                avg(t.amount) as avg_amount,
                stdev(t.amount) as std_amount,
                min(t.timestamp) as first_transaction,
                max(t.timestamp) as last_transaction
            """
            result = self.graph.run(query, customer_id=customer_id)
            return dict(result.data()[0])
        except Exception as e:
            logger.error(f"Error getting customer behavioral patterns: {str(e)}")
            raise

    def get_merchant_risk_score(self, merchant_id: str) -> float:
        """Calculate risk score for a merchant based on transaction patterns."""
        try:
            query = """
            MATCH (m:Merchant {id: $merchant_id})<-[:TO]-(t:Transaction)
            WITH m, t
            RETURN 
                count(t) as transaction_count,
                avg(t.amount) as avg_amount,
                stdev(t.amount) as std_amount,
                count(DISTINCT t.customer_id) as unique_customers
            """
            result = self.graph.run(query, merchant_id=merchant_id)
            data = result.data()[0]
            
            # Simple risk score calculation
            risk_score = (
                (data['std_amount'] / data['avg_amount'] if data['avg_amount'] > 0 else 0) * 0.4 +
                (1 / data['unique_customers'] if data['unique_customers'] > 0 else 1) * 0.3 +
                (data['transaction_count'] / 1000 if data['transaction_count'] > 0 else 0) * 0.3
            )
            
            return min(max(risk_score, 0), 1)  # Normalize between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating merchant risk score: {str(e)}")
            raise

def get_or_create_customer(graph, customer_data):
    matcher = NodeMatcher(graph)
    customer = matcher.match("Customer", id=customer_data['id']).first()
    if customer is None:
        customer = Node("Customer", **customer_data)
        graph.create(customer)
    return customer

def get_or_create_merchant(graph, merchant_data):
    matcher = NodeMatcher(graph)
    merchant = matcher.match("Merchant", id=merchant_data['id']).first()
    if merchant is None:
        merchant = Node("Merchant", **merchant_data)
        graph.create(merchant)
    return merchant

def get_or_create_account(graph, account_data):
    matcher = NodeMatcher(graph)
    account = matcher.match("Account", id=account_data['id']).first()
    if account is None:
        account = Node("Account", **account_data)
        graph.create(account)
    return account 