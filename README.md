AeroDeep Maritime AnalyticsMultimodal Fault Diagnostic & Graph Synthesis System

What is AeroDeep?AeroDeep is an end-to-end industrial AI platform built to tackle a classic offshore engineering challenge: predicting and diagnosing mechanical failures in high-pressure compressor units before they cause catastrophic downtime.Most systems rely entirely on simple threshold alerts on sensor data, which generate high false-alarm rates. 

AeroDeep approaches this by fusing multimodal data streams. It blends millisecond-level time-series sensor telemetry with unstructured text (shift logs, maintenance reports, and OCR-scanned PDFs).By mapping these fused vectors onto a Spatio-Temporal Graph Convolutional Network (ST-GCN) that mirrors the machine’s physical topology, the system goes beyond simple anomaly detection to deliver exact component-level root cause diagnosis and continuous Time-to-Failure ($TTF$) regression analysis.

System Architecture Flow

Stream A (High-Freq Sensors) ──┐
                               ├─► Spatial-Temporal Fusion ─► ST-GCN ─► [ TTF Regression ]
Stream B (Maintenance Logs) ───┘   (Node-Level Mapping)                  [ Multi-Class Fault ]
                                                                                   │
                                                                                   ▼
                                                                        Streamlit UI Console
                                                                        
🏗️ Deep-Dive Project Blueprint

The repository is built around a highly decoupled, clean architecture to maintain separation of concerns between raw ingestion, graph deep learning, and user interface delivery.

aerodeep/
├── milestone1/              # Heterogeneous Ingestion Pipeline
│   ├── stream_a/            # High-frequency sensor telemetry loop
│   │   ├── ingestion.py     # Kafka consumer streaming directly into TimescaleDB
│   │   ├── preprocessor.py  # Sliding windowing, FFT transformations, & normalization
│   │   └── feature_extractor.py  # Dense vector generation for physical nodes
│   ├── stream_b/            # Unstructured maintenance text processing
│   │   ├── pdf_parser.py    # Robust PDF extraction & OCR engine for handwritten logs
│   │   ├── log_cleaner.py   # Industrial acronym expansion & regex noise stripping
│   │   ├── embedder.py      # Domain-specific Industrial-BERT text vectorization
│   │   └── vector_store.py  # Native pgvector indexing interface
│   └── tests/               # Ingestion integration test suites
│
├── milestone2/              # Spatio-Temporal Graph Construction
│   ├── graph/
│   │   ├── schema.py        # Compressor structural adjacency & edge declarations
│   │   ├── builder.py       # Compiling PyG (PyTorch Geometric) HeteroData objects
│   │   └── visualiser.py    # Topo-plot rendering engine for debugging
│   └── fusion/
│       ├── node_fusion.py   # Mathematical alignment of time-series ⊕ text vectors
│       └── stgcn.py         # Spatio-Temporal Graph Convolutional Network architecture
│
├── milestone3/              # Model Alignment, Serving & Dashboard UI
│   ├── model/
│   │   ├── dual_head.py     # Shared GCN backbone splitting to Regression + Classification
│   │   ├── trainer.py       # Production PyTorch Lightning training loop
│   │   └── evaluator.py     # PR-AUC optimization, confusion matrices, & SHAP explanations
│   ├── api/
│   │   ├── main.py          # High-throughput FastAPI inference server
│   │   └── schemas.py       # Strict Pydantic network request/response models
│   └── dashboard/           # Decoupled Streamlit Workspace Panels
│       ├── app.py           # Main application shell & secure session state manager
│       ├── graph_view.py    # Interactive asset topology mapping with API fallback
│       ├── risk_panel.py    # Dynamic metric metrics & continuous risk indicators
│       └── diagnostics.py   # Prescriptive maintenance & root-cause checklist
│
├── configs/                 # Centralized configuration management
│   ├── config.yaml          # Global hyperparameters, DB paths, and network ports
│   └── logging.yaml         # JSON-formatted structured logging for production
│
├── requirements.txt         # Root Python dependency pinning
└── docker-compose.yml       # Orchestration layer for Kafka, Timescale, and Postgres



📈 Roadmap & Milestones

The project is tracked across three strict four-week engineering phases:

Milestone 1: Data Ingestion 
Building the foundations. Setting up Kafka topics, establishing the TimescaleDB layer, and creating the OCR pipeline for unstructured historical logs.

Milestone 2: Graph Synthesis & GCN 
Defining the compressor's physical node connections. Merging context vectors and establishing the spatial-temporal message-passing model architecture.

Milestone 3: Deployment & Dashboard UI
Training the dual-head network using specialized Precision-Recall optimization. Standing up the FastAPI server, building defensive front-end fallbacks, and launching the operational cockpit.


🚀 Quickstart Guide

🛠️ Prerequisites
Make sure your environment has Python 3.10+ installed alongside Docker and Docker Compose.
 Installation & Environment
SetupClone the repository and install the locked operational dependencies:

Bash

git clone https://github.com/your-repo/aerodeep.git
cd aerodeep
pip install -r requirements.txt

 Stand Up Core InfrastructureSpin up the coordinated database and streaming containers in detached mode:Bashdocker-compose up -d
This launches Kafka, TimescaleDB, and PostgreSQL with the pgvector extension preconfigured.3. Kick Off Ingestion ConsumersInitialize the background ingestion loops to capture and process streaming telemetry data:

Bash
# Start processing sensor telemetry streams

python -m milestone1.stream_a.ingestion --config configs/config.yaml

# Start text log processing and embedding generation
python -m milestone1.stream_b.embedder --config configs/config.yaml

 Train the Diagnostic NetworkExecute the training loop to fine-tune the multi-head model weights across your asset histories:
 
 Bash
 
 python -m milestone3.model.trainer --config configs/config.yaml

 Launch the Operations DashboardFire up the responsive, interactive Streamlit monitoring UI locally:
 
 Bash
 
 streamlit run milestone3/dashboard/app.py

 
🛡️ Defensive Engineering & Robustness

The system implements a zero-trust model toward data quality and network states.Schema Drift Guarding: The telemetry pipeline checks incoming streaming formats against runtime structural definitions before executing writes, protecting down-stream models from crashing on broken shapes.Graceful API Fallbacks: If the live graph microservice drops or throws network allocation bugs (such as [Errno 99]), the interface automatically traps the exception, fires an alert notification, and seamlessly switches to rendering locally cached standalone functional topology maps[cite: 2]. This guarantees operators are never left looking at a broken web screen mid-shift.
