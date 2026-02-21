# Time Series ML Demonstration

Two end-to-end time-series machine learning modeling scenarios built on synthetic data:

1. **RMA Freight Forecasting** — GRU recurrent network and Facebook Prophet for shipping weight prediction
2. **Network Event Classification** — K-means clustering with Latent Semantic Analysis (LSA) for unsupervised log categorization

Both pipelines run from data generation through model training, evaluation, and a FastAPI serving layer.

## Blog Post

For a detailed walkthrough of the design decisions, code architecture, and results from this project, read the companion blog post:

**[Someone Else Owns my Best Code, So I Wrote It All Again](https://blog.justintime.ai/rebuilding-ml-projects-time-series/)**: covers the full lifecycle from synthetic data generation through GRU and Prophet model training to the FastAPI serving layer, with code snippets and references throughout.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python scripts/01_generate_data.py
python scripts/02_explore_data.py
python scripts/03_train_rma_model.py
python scripts/03b_train_rma_prophet.py
python scripts/04_train_clustering_model.py
python scripts/05_evaluate_models.py
python scripts/06_export_for_serving.py

# Start API server (optional)
uvicorn mlops.serving.app:app --host 0.0.0.0 --port 8000
```

## Synthetic Data

Both datasets are generated from scratch by `01_generate_data.py` with controlled random seeds for reproducibility. See [`data/README.md`](data/README.md) for full column schemas and statistics.

### RMA Shipping Data (`data/raw/rma_shipping_data.csv`)

~50,000 records over 2 years simulating RMA shipping activity across 5 regions and 15 SKU categories. Embedded patterns include weekly seasonality (lower weekend volume), month/quarter-end spikes, regional growth trends (APAC +8%/year), and urgency-correlated shipping methods.

### Network Events Data (`data/raw/network_events.csv`)

~30,000 events over 30 days with 6 ground-truth behavioral clusters (normal web, normal DB, suspicious scan, auth failure, data exfiltration, maintenance). Each cluster has distinct port ranges, duration profiles, byte volumes, and log message vocabulary for TF-IDF/LSA feature extraction.

## Driver Scripts

| Script | Purpose | Outputs |
|--------|---------|---------|
| `01_generate_data.py` | Generate both synthetic datasets | `data/raw/*.csv` |
| `02_explore_data.py` | Exploratory data analysis and visualization | `outputs/figures/eda_*.png` |
| `03_train_rma_model.py` | Train GRU models (v1 → v3 progressive) | `outputs/models/rma_gru_v*/` |
| `03b_train_rma_prophet.py` | Train Prophet models (v1 basic, v2 with regressors) | `outputs/models/rma_prophet_v*/` |
| `04_train_clustering_model.py` | Train K-means + LSA clustering pipeline | `outputs/models/network_clustering_v1/` |
| `05_evaluate_models.py` | Cross-model evaluation and comparison | `outputs/figures/eval_*.png` |
| `06_export_for_serving.py` | Package models for FastAPI inference | `outputs/models/*/metadata.json` |

## Project Structure

```
time/
├── config/settings.yaml              # Centralized configuration
├── data/
│   ├── raw/                           # Generated synthetic datasets
│   ├── processed/                     # Preprocessed data
│   └── README.md                      # Data dictionary
├── src/
│   ├── data_generation/               # Synthetic data generators
│   ├── preprocessing/                 # Feature engineering pipelines
│   ├── models/                        # Model architectures (GRU, Prophet, K-means, LSA)
│   ├── training/                      # Training pipelines
│   ├── evaluation/                    # Metrics and evaluation
│   ├── visualization/                 # Plotting utilities
│   └── utils/                         # Helper functions
├── mlops/
│   ├── model_registry.py              # Model serialization
│   ├── inference.py                   # Inference engines
│   └── serving/                       # FastAPI application
├── scripts/
│   ├── 01_generate_data.py            # Data generation
│   ├── 02_explore_data.py             # EDA and visualization
│   ├── 03_train_rma_model.py          # GRU model training
│   ├── 03b_train_rma_prophet.py       # Prophet model training
│   ├── 04_train_clustering_model.py   # K-means + LSA training
│   ├── 05_evaluate_models.py          # Model evaluation
│   └── 06_export_for_serving.py       # MLOps preparation
├── tests/
└── outputs/
    ├── figures/                       # Generated visualizations
    ├── models/                        # Saved model artifacts
    └── logs/                          # Training logs
```

---

## Scenario 1: RMA Freight Forecasting

### Business Problem
Unpredictable freight costs for spare parts shipping lead to over-provisioning or costly emergency air shipments.

**Current State:** Naive persistence forecasting (last week's average) yields ~114 kg/day MAE.

**Goal:** Reduce forecast error by 35%+ to enable proactive freight allocation.

### GRU Approach

A stacked GRU recurrent network trained with progressive complexity:

```
Input (30 days × features)
    │
    ▼
Embedding Layers (region, SKU, urgency, method)
    │
    ▼
Stacked GRU (2 layers, 64 units each, dropout=0.3)
    │
    ▼
Layer Normalization
    │
    ▼
Residual Connection (+ avg embeddings)
    │
    ▼
MLP Output (7-day forecast)
```

### Prophet Approach

Facebook Prophet with multiplicative seasonality, trained in two versions: a baseline with automatic seasonality detection, and an enhanced version with exogenous regressors (month-end flags, failure rates, average urgency).

### Progressive Model Improvement

| Version | Changes | MAE (kg) | Improvement |
|---------|---------|----------|-------------|
| Baseline | Naive persistence | 114 | — |
| GRU V1 | Simple GRU, numerical only | 63 | 44% |
| GRU V2 | + Categorical embeddings | 63 | 44% |
| GRU V3 | + Layer norm, residual, dropout | 62 | 46% |
| Prophet V1 | Seasonality decomposition | — | — |
| Prophet V2 | + Exogenous regressors | — | — |

### Key Features
- **Sequence length**: 30 days of history
- **Prediction horizon**: 7 days ahead
- **Categorical embeddings**: Learn semantic relationships
- **Exogenous features**: Repair cycle time, failure rates
- **Early stopping**: Prevents overfitting

---

## Scenario 2: Network Event Classification

### Business Problem
Fragmented logs and inconsistent monitoring made real-time classification and root-cause analysis nearly impossible.

### Solution
An unsupervised learning pipeline combining K-means clustering with Latent Semantic Analysis (LSA) to surface meaningful patterns and classify anomalous activity.

### Pipeline Architecture

```
Network Events
    │
    ├─► Numerical Features ─► StandardScaler
    │
    └─► Log Messages ─► TF-IDF ─► LSA (20 components)
                                    │
                                    ▼
                            Feature Concatenation
                                    │
                                    ▼
                              K-Means Clustering
                                    │
                                    ▼
                          Cluster Interpretation
```

### Discovered Clusters

| Cluster | Characteristics | % of Events |
|---------|-----------------|-------------|
| Normal Web | Short duration, moderate bytes | ~35% |
| Normal DB | Very short, internal traffic | ~20% |
| Suspicious Scan | Tiny packets, many ports | ~10% |
| Auth Failure | Error severity, SSH ports | ~12% |
| Data Exfil | Long duration, high bytes | ~8% |
| Maintenance | Off-hours, internal | ~15% |

### Key Features
- **Automatic K selection**: Elbow method + silhouette score
- **Text analysis**: TF-IDF with n-grams + LSA
- **Anomaly scoring**: Distance to cluster centroid
- **Interpretable results**: Human-readable cluster labels

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### RMA Forecast
```bash
curl -X POST http://localhost:8000/api/v1/rma/forecast \
  -H 'Content-Type: application/json' \
  -d @outputs/models/example_rma_request.json
```

### Network Classification
```bash
curl -X POST http://localhost:8000/api/v1/network/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "source_ip": "10.0.0.1",
    "dest_ip": "192.168.1.100",
    "port": 443,
    "duration_ms": 150.5,
    "bytes_transferred": 8192,
    "protocol": "TCP",
    "log_message": "GET request completed successfully"
  }'
```

---

## Key Technical Decisions

### Why GRU over LSTM?
- Simpler architecture (2 gates vs 3)
- Fewer parameters, faster training
- Comparable performance on moderate sequences

### Why K-means + LSA?
- K-means: Fast, interpretable, scales well
- LSA: Captures semantic relationships in text
- Combined: Multi-modal feature fusion
- No labels needed: Unsupervised discovery

### Why PyTorch?
- More Pythonic API
- Easy debugging with eager execution
- Strong community support
- Flexible for research and production

---

## Model Versioning

Models are saved with full artifacts:

```
# RMA GRU Model
outputs/models/rma_gru_v3/
├── model_weights.pt          # PyTorch state dict
├── preprocessor.joblib       # Encoders and scalers
├── config.json               # Model configuration
├── training_history.json     # Loss curves
└── metadata.json             # Version info

# RMA Prophet Model
outputs/models/rma_prophet_v2/
├── prophet_model.json        # Serialized Prophet model
├── config.json               # Model configuration
└── metadata.json             # Version info

# Clustering Model
outputs/models/network_clustering_v1/
├── clusterer.joblib          # K-means + LSA pipeline
├── preprocessor.joblib       # TF-IDF, scalers
├── cluster_interpretations.json
└── metadata.json
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- scikit-learn 1.3+
- Prophet
- FastAPI 0.100+
- See `requirements.txt` for full list

### Hardware
- Tested on M2 MacBook Pro
- Memory: ~4GB peak

---

## License

This project was developed and released for educational and demonstration purposes.
