# AeroDeep Maritime Analytics
## Multimodal Fault Diagnostic & Graph Synthesis System

**Client:** AeroDeep Maritime Analytics  
**Project:** Offshore Asset Fault Diagnosis — Gulf of Guinea Operations  
**Budget:** $85,000 USD | **Timeline:** 12 Weeks

---

## System Overview

This system fuses millisecond-level sensor telemetry with unstructured maintenance logs and shift reports to deliver precise root-cause fault diagnosis for offshore compressor units. It moves beyond anomaly detection into **component-level diagnosis with time-to-failure prediction**.

### Architecture
```
Stream A (Sensor Telemetry) ──┐
                               ├─► Node-Level Fusion ─► ST-GCN ─► [TTF Score | Root Cause]
Stream B (Maintenance Logs) ──┘                                         │
                                                                         ▼
                                                              Interactive Dashboard
```

---

## Project Structure

```
aerodeep/
├── milestone1/              # Heterogeneous Ingestion Pipeline
│   ├── stream_a/            # Sensor telemetry processing
│   │   ├── ingestion.py     # Kafka consumer + TimescaleDB writer
│   │   ├── preprocessor.py  # Windowing, FFT, normalisation
│   │   └── feature_extractor.py  # Dense vector generation
│   ├── stream_b/            # Unstructured text processing
│   │   ├── pdf_parser.py    # PDF/scan OCR extraction
│   │   ├── log_cleaner.py   # Noise removal, abbreviation expansion
│   │   ├── embedder.py      # Industrial BERT / LLM embedding
│   │   └── vector_store.py  # pgvector interface
│   └── tests/
│       ├── test_stream_a.py
│       └── test_stream_b.py
│
├── milestone2/              # Spatio-Temporal Graph Construction
│   ├── graph/
│   │   ├── schema.py        # Compressor graph topology definition
│   │   ├── builder.py       # PyG HeteroData construction
│   │   └── visualiser.py    # Graph topology plotting
│   ├── fusion/
│   │   ├── node_fusion.py   # Time-series ⊕ text embedding fusion
│   │   └── stgcn.py         # ST-GCN model architecture
│   └── tests/
│       └── test_graph.py
│
├── milestone3/              # Model Alignment & Dashboard
│   ├── model/
│   │   ├── dual_head.py     # TTF regression + fault classification heads
│   │   ├── trainer.py       # PyTorch Lightning training loop
│   │   └── evaluator.py     # Metrics, confusion matrix, SHAP
│   ├── api/
│   │   ├── main.py          # FastAPI inference server
│   │   └── schemas.py       # Pydantic request/response models
│   ├── dashboard/
│   │   ├── app.py           # Streamlit multipage dashboard
│   │   ├── graph_view.py    # Live interactive asset graph
│   │   ├── risk_panel.py    # Risk migration visualisation
│   │   └── diagnostics.py   # Root cause explanation panel
│   └── tests/
│       └── test_api.py
│
├── configs/
│   ├── config.yaml          # All runtime configuration
│   └── logging.yaml         # Structured logging config
│
├── docs/
│   └── architecture.md      # Detailed system design notes
│
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## Milestones

| Milestone | Scope | Weeks | Budget |
|-----------|-------|-------|--------|
| M1 | Heterogeneous Ingestion Pipeline | 1–4 | $25,000 |
| M2 | Spatio-Temporal Graph Construction | 5–8 | $35,000 |
| M3 | Model Alignment & Diagnostic Dashboard | 9–12 | $25,000 |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker + Docker Compose
- PostgreSQL 15+ with pgvector extension

### Installation
```bash
pip install -r requirements.txt
docker-compose up -d   # starts Kafka, TimescaleDB, pgvector
```

### Run Ingestion Pipeline
```bash
python -m milestone1.stream_a.ingestion --config configs/config.yaml
python -m milestone1.stream_b.embedder --config configs/config.yaml
```

### Train Model
```bash
python -m milestone3.model.trainer --config configs/config.yaml
```

### Launch Dashboard
```bash
streamlit run milestone3/dashboard/app.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Streaming | Apache Kafka |
| Time-series DB | TimescaleDB (PostgreSQL) |
| Vector Store | pgvector on PostgreSQL |
| NLP Embedding | RoBERTa-base (industrial fine-tune) |
| Graph Framework | PyTorch Geometric |
| Training | PyTorch Lightning + Weights & Biases |
| Inference API | FastAPI |
| Dashboard | Streamlit |
| Containerisation | Docker + Docker Compose |
