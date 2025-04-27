# Real-Time Financial Behavior Modeling and Anomaly Detection System

A comprehensive system for modeling customer financial behavior, detecting micro-patterns, and identifying anomalies using machine learning models, specifically designed for the Indian financial ecosystem.

## Features

- Customer Financial Twin creation
- Micro-pattern detection (timing, merchant relationships, spending velocity)
- Real-time anomaly detection
- Graph-based relationship modeling using Neo4j
- RESTful API interface using FastAPI
- Compliance with Indian financial regulations (RBI DPDP Act, Payment Aggregator Guidelines)

## Project Structure

```
financial_risk/
├── data/                   # Data storage and processing
│   ├── raw/               # Raw transaction data
│   └── processed/         # Processed features
├── models/                # ML model implementations
│   ├── anomaly/          # Anomaly detection models
│   └── clustering/       # Behavioral clustering models
├── api/                   # FastAPI application
├── graph/                 # Neo4j graph database models
├── utils/                 # Utility functions
├── config/               # Configuration files
└── tests/                # Test suite
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd financial-risk
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Neo4j database:
- Install Neo4j Desktop or use Neo4j Aura
- Create a new database
- Update the connection details in `config/database.yaml`

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the API server:
```bash
uvicorn api.main:app --reload
```

2. Access the API documentation:
- Open http://localhost:8000/docs in your browser

## Development

### Running Tests
```bash
pytest
```

### Code Style
```bash
black .
flake8
mypy .
```

## Compliance and Security

- All customer data is anonymized and encrypted
- API access is restricted and secured
- Regular VAPT (Vulnerability Assessment and Penetration Testing) performed
- Compliant with RBI regulations and DPDP Act

## License

[Your License]

## Contributing

[Contribution Guidelines] 