# AeroDeep: Predictive Diagnostics for Offshore Compressors

> **Millisecond-level sensor telemetry + unstructured maintenance logs → component-level fault diagnosis + time-to-failure prediction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

AeroDeep is an end-to-end industrial AI platform that moves beyond anomaly detection to deliver **component-level root-cause diagnosis and predictive maintenance** for high-pressure offshore compressor units.

### The Challenge
Offshore compressor failures are costly—both in downtime and safety risk. Traditional approaches rely on reactive maintenance or coarse-grained anomaly detection that fails to pinpoint *which component* will fail *when*.

### The Solution
AeroDeep fuses **multimodal heterogeneous data**:
- **Stream A**: High-frequency time-series sensor telemetry (vibration, temperature, pressure)
- **Stream B**: Unstructured maintenance logs, shift reports, and OCR-scanned PDFs

Through a **Spatio-Temporal Graph Convolutional Network (ST-GCN)**, AeroDeep learns the physical topology of each compressor and predicts:
1. **Time-to-Failure (TTF)** — regression output
2. **Component Fault Type** — multi-class classification output

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Multimodal Data Ingestion                      │
├─────────────────────────────────────────────────────────────────┤
│  Stream A (Sensors)              │  Stream B (Logs & Text)      │
│  • Kafka consumers               │  • PDF parser + OCR          │
│  • FFT & sliding windows         │  • Industrial-BERT embedding │
│  • Feature extraction            │  • pgvector indexing         │
└─────────────────┬────────────────┬──────────────────────────────┘
                  │                │
                  ▼                ▼
        ┌────────────────────────────────┐
        │  Spatio-Temporal Fusion        │
        │  • Node-level alignment        │
        │  • Heterogeneous graph schema  │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  ST-GCN Neural Network         │
        │  • Shared GCN backbone         │
        │  • Dual-head output            │
        │    ├─ TTF Regression           │
        │    └─ Fault Classification     │
        └────────────┬───────────────────┘
                     │
        ┌────────────┴───────────────────┐
        │    Production Services         │
        ├────────────┬───────────────────┤
        │ FastAPI   │  Streamlit         │
        │ Inference │  Dashboard         │
        └───────────┴───────────────────┘
```

---

## Project Structure

```
aerodeep/
├── milestone1/                  # Heterogeneous Data Ingestion
│   ├── stream_a/               # High-frequency sensor pipeline
│   │   ├── ingestion.py        # Kafka → TimescaleDB consumer
│   │   ├── preprocessor.py     # FFT, windowing, normalization
│   │   └── feature_extractor.py # Dense vector generation
│   ├── stream_b/               # Unstructured text processing
│   │   ├── pdf_parser.py       # OCR & PDF extraction
│   │   ├── log_cleaner.py      # Acronym expansion & noise removal
│   │   ├── embedder.py         # Industrial-BERT vectorization
│   │   └── vector_store.py     # pgvector indexing
│   └── tests/                  # Integration test suites
│
├── milestone2/                  # Graph Synthesis & GCN
│   ├── graph/
│   │   ├── schema.py           # Compressor topology definition
│   │   ├── builder.py          # PyTorch Geometric HeteroData
│   │   └── visualizer.py       # Graph rendering & debugging
│   └── fusion/
│       ├── node_fusion.py      # Time-series ⊕ text alignment
│       └── stgcn.py            # ST-GCN model architecture
│
├── milestone3/                  # Model & Deployment
│   ├── model/
│   │   ├── dual_head.py        # Shared backbone + dual heads
│   │   ├── trainer.py          # PyTorch Lightning training loop
│   │   └── evaluator.py        # PR-AUC & SHAP explainability
│   ├── api/
│   │   ├── main.py             # FastAPI inference server
│   │   └── schemas.py          # Pydantic validation models
│   └── dashboard/
│       ├── app.py              # Streamlit main shell
│       ├── graph_view.py       # Interactive topology
│       ├── risk_panel.py       # Real-time risk indicators
│       └── diagnostics.py      # Root-cause checklist
│
├── configs/                     # Configuration management
│   ├── config.yaml             # Global hyperparameters
│   └── logging.yaml            # Structured logging
│
├── tests/                       # Cross-module test suite
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Orchestration
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Docker & Docker Compose**
- **Git**

### 1. Clone & Install

```bash
git clone https://github.com/obaloluwakoyi/aerodeep.git
cd aerodeep
pip install -r requirements.txt
```

### 2. Start Infrastructure

```bash
docker-compose up -d
```

This provisions:
- **Kafka** — streaming message broker
- **TimescaleDB** — time-series telemetry storage
- **PostgreSQL** — metadata & pgvector embeddings

### 3. Run Ingestion Pipelines

```bash
# Terminal 1: Start sensor telemetry processor
python -m milestone1.stream_a.ingestion --config configs/config.yaml

# Terminal 2: Start maintenance log embedder
python -m milestone1.stream_b.embedder --config configs/config.yaml
```

### 4. Train the Model

```bash
python -m milestone3.model.trainer --config configs/config.yaml
```

### 5. Launch Dashboard

```bash
streamlit run milestone3/dashboard/app.py
```

Navigate to `http://localhost:8501` to access the operations console.

---

## 📋 Development Roadmap

### **Milestone 1: Data Ingestion** 
- ✅ Kafka topic setup & consumer logic
- ✅ TimescaleDB schema design
- ✅ OCR pipeline for historical PDFs
- ✅ Industrial-BERT embeddings

### **Milestone 2: Graph Synthesis & ST-GCN** 
- ✅ Compressor topology modeling
- ✅ Node-level feature fusion
- ✅ Spatio-temporal GCN architecture
- ✅ Graph visualization tools

### **Milestone 3: Production & Dashboard** 
- ✅ Dual-head model training (TTF + Classification)
- ✅ FastAPI inference server
- ✅ Streamlit operations dashboard
- ✅ SHAP explainability & diagnostics

---

## 🛡️ Design Principles

### Zero-Trust Data Quality
Every incoming data point is validated against runtime schema definitions. Drift is detected and logged immediately.

### Defensive API Design
The Streamlit frontend includes fallback logic—if the inference API is unavailable, historical risk scores are displayed.

### Explainability-First
SHAP values and attention visualizations are embedded into every prediction to support operator decision-making.

### Clean Architecture
Strict separation between ingestion, graph construction, model training, and UI layers ensures independent testing and deployment.

---

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Database connections
timescaledb:
  host: localhost
  port: 5432
  database: telemetry

# Kafka topics
kafka:
  brokers:
    - localhost:9092
  sensor_topic: compressor.telemetry
  batch_size: 1024

# Model training
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  ttf_weight: 0.6
  classification_weight: 0.4
```

---

## 📊 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Streaming** | Apache Kafka | Real-time telemetry ingestion |
| **Time-Series DB** | TimescaleDB | High-cardinality sensor data |
| **Embeddings** | pgvector + Industrial-BERT | Text vectorization & similarity search |
| **Graph ML** | PyTorch Geometric | Heterogeneous graph operations |
| **GCN Model** | PyTorch Lightning | Distributed training & reproducibility |
| **API Server** | FastAPI | High-throughput inference |
| **Dashboard** | Streamlit | Interactive operations console |
| **Explainability** | SHAP | Feature importance attribution |

---

## 📦 Dependencies

See `requirements.txt` for the complete dependency list. Key packages:

```
torch>=2.0
torch-geometric>=2.4
pytorch-lightning>=2.1
fastapi>=0.100
streamlit>=1.28
sqlalchemy>=2.0
pgvector>=0.2
kafka-python>=2.0
```

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v --cov=aerodeep
```

Run integration tests only:

```bash
pytest tests/integration/ -v
```

---

## 📖 Documentation

- [Data Ingestion Guide](docs/ingestion.md)
- [Graph Schema Reference](docs/graph_schema.md)
- [Model Training Guide](docs/training.md)
- [API Documentation](docs/api.md)
- [Dashboard User Guide](docs/dashboard.md)

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

Please ensure:
- Code passes `black` and `isort` formatting
- All tests pass (`pytest`)
- New features include docstrings and tests

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Balolowakoyi**  
[GitHub](https://github.com/obaloluwakoyi) | [Email](obaloluw@gmail.com)

---

## 🙏 Acknowledgments

Built with:
- PyTorch & PyTorch Geometric community
- Streamlit framework
- Industrial ML best practices from the predictive maintenance community

---

## 📞 Support & Questions

For issues, feature requests, or questions:
- 📝 [Open an Issue](https://github.com/obaloluwakoyi/aerodeep/issues)
- 💬 [GitHub Discussions](https://github.com/obaloluwakoyi/aerodeep/discussions)

---

<div align="center">

**AeroDeep**: Where Industrial Data Becomes Predictive Intelligence



</div>
