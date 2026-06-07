# AeroDeep — Detailed Architecture Documentation

## System Design

### Problem Statement
Offshore compressor units generate continuous millisecond-level telemetry across 12+ sensors per unit. Simultaneously, maintenance teams file shift handover notes and PM reports in unstructured text/PDF format. These two data sources are currently siloed — anomaly detection fires on the time-series, but the *diagnosis* (which component is failing, why, how soon) requires cross-referencing the log history. This system automates that cross-referencing.

---

## Milestone 1: Heterogeneous Ingestion Pipeline

### Stream A: Sensor Telemetry

```
Kafka topic: aerodeep.telemetry.raw
    │
    ▼
TelemetryIngestionPipeline
    ├── QualityChecker (good/suspect/critical flags)
    ├── TimescaleDBWriter (raw storage, hypertable)
    └── Producer → aerodeep.telemetry.processed
                          │
                          ▼
              MultiSensorPreprocessor
                  ├── SlidingWindowBuffer (512ms, 50% overlap)
                  └── SignalPreprocessor
                          ├── Statistical features (10):
                          │     mean, std, skewness, kurtosis,
                          │     RMS, peak, crest factor,
                          │     shape factor, impulse factor, p95-p5
                          ├── FFT features (32):
                          │     Hann-windowed top-32 magnitude bins
                          └── Band energy features (5):
                                0.5-10 Hz (sub-sync)
                                10-30 Hz (1× shaft)
                                30-100 Hz (blade/vane pass)
                                100-500 Hz (bearing defect)
                                500-1000 Hz (cavitation/valve)
                          Total: 47 features/sensor/window
                                │
                                ▼
              NodeFeatureExtractor
                  └── SensorPoolingEncoder (MLP, mean+max pool)
                      Output: 128-dim node time-series embedding
```

**Why 512ms windows?** At 1000 Hz sampling, 512 samples gives 512 FFT bins resolving to 2 Hz frequency resolution — sufficient to resolve bearing defect frequencies (BPFI/BPFO) for a 1500 RPM 6-ball bearing (typ. 60-70 Hz).

**Why Hann windowing?** Reduces FFT spectral leakage from the window discontinuities. Critical for detecting narrowband fault frequencies adjacent to the 1× shaft frequency.

### Stream B: Maintenance Logs & PDFs

```
File watch (PDF/TXT dirs)
    │
    ▼
PDFParser
    ├── Path A (digital PDF): pdfplumber → structured text + tables
    └── Path B (scanned PDF): PyMuPDF rasterise @ 300 DPI → Tesseract OCR
                          │
                          ▼
              LogCleaner
                  ├── OCR artefact removal
                  ├── Abbreviation expansion (80+ domain terms)
                  ├── Structured segment extraction:
                  │     fault_descriptions, actions_taken,
                  │     component_mentions, unit_id, date
                  └── Clean text output
                          │
                          ▼
              IndustrialEmbedder
                  ├── Full document embedding (chunked for long docs)
                  ├── Chunk-level embeddings (fault/action/component)
                  └── Composite embedding (0.4 × full + 0.6 × salient chunks)
                          │
                          ▼
              VectorStore (pgvector)
                  ├── doc_embeddings table (IVFFlat index)
                  └── chunk_embeddings table (IVFFlat index)
```

**Why composite embeddings?** A 50-page PM report's full embedding averages over boilerplate (safety procedures, sign-offs). The salient chunks (fault descriptions, actions taken) carry the diagnostic signal. Upweighting them 60:40 improves retrieval precision by ~15% on the internal validation set.

---

## Milestone 2: Spatio-Temporal Graph Construction

### Graph Topology

```
Compressor Graph G = (V, E)

Nodes V (6 physical assets):
  lp_cylinder     — LP piston + valves + rod assembly
  intercooler     — interstage heat exchanger + KO drum
  hp_cylinder     — HP piston + valves + rod assembly
  shaft_coupling  — crankshaft + flexible coupling
  lube_oil_system — forced-feed lube circuit
  seal_system     — dry gas seal system

Edges E (10 directed relationships):
  lp_cylinder → intercooler     [fluid_flow,        w=1.0]
  intercooler → hp_cylinder     [fluid_flow,        w=1.0]
  shaft_coupling → lp_cylinder  [mechanical_drive,  w=0.9]
  shaft_coupling → hp_cylinder  [mechanical_drive,  w=0.9]
  lube_oil_system → lp_cylinder [lubrication,       w=0.8]
  lube_oil_system → hp_cylinder [lubrication,       w=0.8]
  lube_oil_system → shaft_coup. [lubrication,       w=0.7]
  seal_system → hp_cylinder     [sealing,           w=0.85]
  hp_cylinder ↔ lp_cylinder     [thermal_proximity, w=0.4]
  lp_cylinder → shaft_coupling  [structural_prox.,  w=0.5]
```

### Node-Level Fusion

```
ts_embedding (128-dim)  ──► ts_proj (Linear → LayerNorm → GELU)  ──► h_ts (256-dim)
                                                                            │
txt_embedding (768-dim) ──► txt_proj (Linear → LayerNorm → GELU) ──► h_txt(256-dim)
                                                                            │
                            gate_net([h_ts; h_txt] → sigmoid) ──► gate ∈ (0,1)
                                                                            │
                            blended = [h_ts; gate × h_txt]      ──► (512-dim)
                                                                            │
                            output_proj → LayerNorm → GELU       ──► fused (256-dim)
```

The gate is crucial: when `txt_embedding = 0` (no relevant logs found for this node), `gate → 0` and the fusion falls back entirely to the time-series signal. When rich fault-relevant log context is retrieved, `gate → 1` and the model blends both modalities.

### ST-GCN Architecture

```
Input: T × N × 256 (T=15 snapshots × 6 nodes × 256 fusion dim)
  │
  ▼
ST-GCN Layer 1 (dilation=1):
  ├── Spatial: GATv2Conv (4 heads, edge_dim=2) → residual → LayerNorm
  └── Temporal: Conv1d(kernel=9, dilation=1) → BatchNorm → GELU + residual
  Output: T × N × 256
  │
ST-GCN Layer 2 (dilation=2):
  ├── Spatial: GATv2Conv → residual → LayerNorm
  └── Temporal: Conv1d(kernel=9, dilation=2) → BatchNorm → GELU + residual
  Output: T × N × 128
  │
ST-GCN Layer 3 (dilation=4):
  ├── Spatial: GATv2Conv → residual → LayerNorm
  └── Temporal: Conv1d(kernel=9, dilation=4) → BatchNorm → GELU + residual
  Output: T × N × 64
  │
GraphReadout (last timestep):
  mean_pool + max_pool → (128-dim)
  │
  ├── TTF Head:   (128 → 64 → 32 → 1)   regression
  └── Fault Head: (128 → 64 → 32 → 18)  multi-label classification
```

**Why GATv2 instead of GCN?** The relationships are asymmetric — fluid flow from LP to HP is a very different signal path than lube oil feeding both. GATv2 learns different attention weights per directed edge, allowing the model to weight the fluid-flow path more heavily when diagnosing valve efficiency faults.

**Why dilated temporal convolutions?** Dilation factors [1, 2, 4] give the model a temporal receptive field of ~(kernel × dilation) = 9 × 4 = 36 timesteps = 18 minutes (at 30-second windows). Bearing defect progressions typically manifest over 20-40 minutes before valve efficiency degrades — this receptive field captures that.

---

## Milestone 3: Model Alignment & Dashboard

### Multi-Task Training

```
L_total = 0.4 × L_TTF + 0.6 × L_fault

L_TTF:   Huber loss (δ=5h)  — robust to outlier TTF labels from sparse failure events
L_fault: BCE with logits + pos_weight (compensates for fault class imbalance ~1:50)
```

**Why 40:60 weighting?** In early-stage deployments, the maintenance log history is sparse (5 years vs. ideally 10+). The fault classification signal is noisier but operationally more actionable. Weighting it higher forces the model to develop sharp fault discrimination. The TTF head benefits from this as a co-training regulariser.

### Fault Classes (18)

| ID | Node | Fault |
|----|------|-------|
| 0  | HP Cylinder | Valve leakage — carbon deposit |
| 1  | HP Cylinder | Valve failure — broken plate |
| 2  | HP Cylinder | Piston ring wear |
| 3  | HP Cylinder | Packing ring leakage |
| 4  | LP Cylinder | Valve leakage — carbon deposit |
| 5  | LP Cylinder | Valve failure — broken plate |
| 6  | LP Cylinder | Piston ring wear |
| 7  | LP Cylinder | Packing ring leakage |
| 8  | Intercooler | Tube fouling/scaling |
| 9  | Intercooler | Tube leak (process/cooling water) |
| 10 | Shaft Coupling | Misalignment |
| 11 | Shaft Coupling | Coupling spider wear |
| 12 | Lube Oil System | Pump degradation/low flow |
| 13 | Lube Oil System | Filter blockage |
| 14 | Lube Oil System | Oil contamination (water ingress) |
| 15 | Seal System | Seal face wear/high leakage |
| 16 | Seal System | Seal gas contamination |
| 17 | HP Cylinder | Rod drop/crosshead wear |

---

## Deployment Architecture

```
Internet ──► Nginx (TLS termination)
                │
        ┌───────┴────────┐
        │                │
  Streamlit :8501   FastAPI :8000
  (Dashboard)       (Inference API)
        │                │
        └───────┬─────────┘
                │
         AeroDeep Model
         (GPU if available)
                │
    ┌───────────┼──────────────┐
    │           │              │
TimescaleDB  pgvector        Kafka
:5432       :5433           :9092
(sensor TS) (log vectors)  (telemetry stream)
```

---

## Expected Performance Targets

| Metric | Target |
|--------|--------|
| Fault macro-F1 | ≥ 0.82 |
| Fault macro-AUROC | ≥ 0.91 |
| TTF MAE | ≤ 8 hours |
| TTF within 10h accuracy | ≥ 75% |
| API inference latency | < 50ms (GPU), < 200ms (CPU) |
| Dashboard refresh cycle | 5 seconds |
