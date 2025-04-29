import json
import uuid
import random
from datetime import datetime, timedelta

# Generate random UUIDs for all IDs to ensure uniqueness
def generate_transaction_data(num_transactions=10):
    transactions = []
    
    # Categories with risk levels
    high_risk_categories = ["GAMBLING", "CRYPTO", "TRANSFER", "LUXURY"]
    normal_categories = ["RETAIL", "FOOD", "GROCERY", "UTILITY"]
    
    # Current time for timestamps
    now = datetime.now()
    
    # Generate some normal transactions
    for i in range(5):
        customer_id = str(uuid.uuid4())
        account_id = str(uuid.uuid4())
        
        # Normal transaction - smaller amount, normal category, business hours
        tx_time = now - timedelta(minutes=random.randint(60, 240))
        tx_time = tx_time.replace(hour=random.randint(9, 17))  # Business hours
        
        transaction = {
            "id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "account_id": account_id,
            "merchant_id": str(uuid.uuid4()),
            "amount": round(random.uniform(10, 500), 2),  # Normal amount
            "timestamp": tx_time.isoformat(),
            "category": random.choice(normal_categories),
            "description": f"Normal transaction {i+1}"
        }
        transactions.append(transaction)
    
    # Generate suspicious transactions (same customer/account with high-risk behavior)
    suspicious_customer_id = str(uuid.uuid4())
    suspicious_account_id = str(uuid.uuid4())
    
    for i in range(5):
        # Suspicious transaction pattern - large amount, high-risk category, unusual hours
        tx_time = now - timedelta(minutes=random.randint(5, 30))
        if random.choice([True, False]):
            tx_time = tx_time.replace(hour=random.randint(0, 4))  # Late night
        else:
            tx_time = tx_time.replace(hour=random.randint(22, 23))  # Late night
        
        # Very large amount for obviously suspicious transactions
        suspicious_amount = round(random.uniform(8000, 20000), 2)
        
        transaction = {
            "id": str(uuid.uuid4()),
            "customer_id": suspicious_customer_id,  # Same customer for related suspicious transactions
            "account_id": suspicious_account_id,    # Same account for related suspicious transactions
            "merchant_id": str(uuid.uuid4()),
            "amount": suspicious_amount,
            "timestamp": tx_time.isoformat(),
            "category": random.choice(high_risk_categories),
            "description": f"Highly suspicious transaction {i+1}"
        }
        transactions.append(transaction)
    
    return transactions

# Generate transactions and save to JSON file
if __name__ == "__main__":
    transactions = generate_transaction_data()
    
    # Print summary of transactions
    normal_count = sum(1 for tx in transactions if float(tx['amount']) < 1000)
    suspicious_count = len(transactions) - normal_count
    print(f"Generated {len(transactions)} transactions:")
    print(f"- {normal_count} normal transactions")
    print(f"- {suspicious_count} suspicious transactions (high amount, high-risk category)")
    
    # Save to file
    with open("test_transactions.json", "w") as f:
        json.dump(transactions, f, indent=2)
    
    print("Saved transactions to test_transactions.json") 