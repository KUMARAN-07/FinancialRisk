# Financial Risk API

A FastAPI-based backend for financial transaction risk analysis, anomaly detection, and behavioral monitoring.

## Features

- **Real-time Transaction Processing**: Process and analyze financial transactions in real-time.
- **Anomaly Detection**: Using Isolation Forest ML models to detect unusual transactions.
- **Behavioral Analysis**: Monitor customer behavior patterns and risk scoring.
- **Graph Database**: Uses Neo4j to model the complex relationships between customers, accounts, merchants, and transactions.
- **WebSocket Support**: Real-time updates for dashboards and monitoring systems.

## Setup Instructions

### Prerequisites

- Python 3.9+
- Neo4j Database (local or remote)
- pip (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/financial-risk-api.git
   cd financial-risk-api
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Neo4j**:
   - Install Neo4j from [https://neo4j.com/download/](https://neo4j.com/download/)
   - Start the Neo4j service
   - Create a new database named "FinancialRisk"
   - Update the credentials in `financial_risk/config/config.yaml`

### Running the API

1. **Start the API server**:
   ```bash
   uvicorn financial_risk.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the API documentation**:
   - OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## API Endpoints

- **POST /dev/generate_and_train**: Generate synthetic data and train models.
- **POST /transactions/process**: Process a batch of transactions.
- **GET /customers/{customer_id}/behavior**: Get customer behavioral patterns.
- **GET /merchants/{merchant_id}/risk**: Get merchant risk score.
- **POST /models/train**: Retrain the ML models.
- **GET /health**: API health check.
- **WebSocket /ws**: Real-time transaction updates.

## WebSocket Usage

Connect to the WebSocket endpoint to receive real-time transaction updates:

```javascript
// Browser example
const socket = new WebSocket("ws://localhost:8000/ws");

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Received transactions:", data);
  // Update UI with transaction data
};

socket.onclose = (event) => {
  console.log("WebSocket connection closed");
};
```

## Troubleshooting

- **Neo4j Connection Issues**: 
  - Ensure Neo4j is running
  - Check credentials in config.yaml
  - Verify network connectivity/firewall settings
  - The API will start even if Neo4j is unavailable and will provide demo data via WebSocket

- **API Not Starting**:
  - Check logs for errors
  - Verify port 8000 is not in use

## License

MIT

## Contact

For support or questions, please contact [your.email@example.com](mailto:your.email@example.com). 