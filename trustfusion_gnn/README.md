# TrustFusion-GNN

TrustFusion-GNN is a synthetic-data-driven research prototype for trustworthy multi-sensor fusion in smart agriculture. It includes four main capabilities:

1. Generate agricultural sensor data with injected faults.
2. Train a trust-aware graph neural network on that data.
3. Run inference and robustness evaluation.
4. Analyze the quality of the generated synthetic dataset.

## Project Workflow

The typical workflow is:

1. Define sensor topology and system configuration.
2. Generate clean and faulty synthetic data.
3. Export train/validation/test splits into `simulated_data/`.
4. Analyze data quality, coverage, and correlations.
5. Train the model either from freshly simulated data or from exported `.npz` files.
6. Evaluate fusion accuracy, anomaly detection, robustness, and per-channel error.

## Execution Entry Points

### `main.py`

Interactive demo entry point for the full prototype.

It can:

1. Show model architecture.
2. Demonstrate data simulation.
3. Demonstrate graph construction.
4. Run a lightweight training flow.
5. Run inference on sample windows.
6. Run robustness tests.

Use it when you want an end-to-end demo of the original prototype behavior.

### `generate_simulated_dataset.py`

Generates larger synthetic train/validation/test datasets and writes them into `simulated_data/`.

Outputs include:

1. `.npz` dataset files.
2. Per-split summary JSON files.
3. Per-split fault metadata JSON files.
4. A manifest file describing the generation run.

Use it when you want reproducible synthetic datasets for repeated experiments.

### `train_on_simulated_data.py`

Loads pre-generated `.npz` files from `simulated_data/` and trains the model directly from those files.

It also:

1. Runs lightweight dataset reasonableness checks.
2. Uses the existing `Trainer` class.
3. Writes run logs and training summaries.
4. Outputs overall and per-channel metrics.

Use it when you want stable, repeatable training from fixed data splits.

### `analyze_simulated_data.py`

Builds a richer quality report for generated data.

It produces:

1. A Markdown quality report.
2. A JSON quality report.
3. Sensor distribution charts.
4. Sensor correlation heatmap.
5. Fault coverage chart.

Use it when you want to decide whether the simulated data is realistic and balanced enough to support training.

## Core Files and Their Roles

### Configuration and Data Definition

### `config.py`

Defines the full agricultural system configuration:

1. Sensor metadata.
2. Sensor ranges and noise levels.
3. Model hyperparameters.
4. Physical constraints.
5. Thresholds and training settings.

### `data_structures.py`

Defines the main typed containers used across the system, including:

1. System inputs.
2. System outputs.
3. Intermediate stage outputs.
4. Application-layer fusion results.

### Data Simulation and Preprocessing

### `data_simulator.py`

Generates synthetic sensor sequences and injects faults.

Main responsibilities:

1. Generate clean agricultural sensor signals.
2. Inject fault types such as drift, noise, spike, bias, stuck-at, and random values.
3. Build fusion targets.
4. Build credibility targets and fault masks.
5. Create PyTorch dataloaders from in-memory simulated data.

### `simulated_dataset_loader.py`

Loads pre-generated `.npz` datasets into PyTorch datasets and dataloaders.

Use it when training from exported data instead of regenerating everything in memory.

### `normalization.py`

Handles min-max normalization and denormalization for model inputs, outputs, and uncertainty values.

### Graph and Model Definition

### `graph_builder.py`

Builds the graph structure used by the GNN.

It supports:

1. Distance-based adjacency.
2. Sensor-type adjacency.
3. ESP32-group adjacency.
4. Combined adjacency.
5. K-nearest-neighbor graphs.

### `models/stage1_feature.py`

Implements Stage 1 of the model:

1. Temporal encoding.
2. Statistical feature extraction.
3. Initial trust-score estimation.

### `models/stage2_graph.py`

Implements Stage 2 of the model:

1. Trust-aware graph attention.
2. Multi-hop message passing.
3. Learned adjacency refinement.
4. Time-varying trust propagation.

### `models/stage3_fusion.py`

Implements Stage 3 of the model:

1. Trust refinement.
2. Trust-weighted fusion.
3. Uncertainty estimation.
4. Anomaly detection.
5. System-confidence estimation.

### `models/trustfusion_gnn.py`

Wraps Stage 1, Stage 2, and Stage 3 into the full TrustFusion-GNN model.

### Training, Evaluation, and Inference

### `trainer.py`

Contains the training loop and validation logic.

Current responsibilities:

1. Train one epoch.
2. Evaluate on validation data.
3. Track best model by validation loss.
4. Optionally write training logs and training history to disk.

### `losses.py`

Defines the composite loss used during training, including:

1. Fusion loss.
2. Trust-score loss.
3. Anomaly loss.
4. Consistency loss.
5. Uncertainty calibration loss.

### `metrics.py`

Defines evaluation metrics.

It now reports:

1. Overall MAE.
2. Overall RMSE.
3. Overall MAPE.
4. Per-channel MAE for `temperature`, `humidity`, `soil_moisture`, and `light`.
5. Per-channel RMSE for `temperature`, `humidity`, `soil_moisture`, and `light`.
6. Anomaly detection metrics.
7. Trust-score and system-confidence metrics.

### `inference.py`

Implements runtime inference and robustness evaluation.

It supports:

1. Single-step buffering into full windows.
2. Full-window inference.
3. Conversion of raw model outputs into application-friendly results.
4. Robustness evaluation under varying anomaly ratios.

## Generated Data and Reports

### `simulated_data/`

Stores generated datasets and derived artifacts.

Typical contents:

1. `train.npz`, `val.npz`, `test.npz`
2. `*_summary.json`
3. `*_faults.json`
4. `manifest.json`
5. `analysis/` quality reports and SVG charts
6. `training_runs/` training logs, histories, and summaries

## Common Commands

Generate a larger dataset:

```bash
/opt/anaconda3/envs/bohan/bin/python generate_simulated_dataset.py \
  --output-dir simulated_data \
  --train-samples 3000 \
  --val-samples 600 \
  --test-samples 600 \
  --fault-ratio 0.35 \
  --seed 42
```

Train from exported data:

```bash
/opt/anaconda3/envs/bohan/bin/python train_on_simulated_data.py \
  --data-dir simulated_data \
  --epochs 5 \
  --patience 3 \
  --batch-size 64
```

Analyze dataset quality:

```bash
/opt/anaconda3/envs/bohan/bin/python analyze_simulated_data.py \
  --data-dir simulated_data \
  --output-dir simulated_data/analysis
```

Run the original interactive demo:

```bash
python main.py
```

## Current Practical Interpretation

At this stage, the project is best understood as a synthetic-data research pipeline.

The most stable path is:

1. Generate data with `generate_simulated_dataset.py`.
2. Review `simulated_data/analysis/quality_report.md`.
3. Train with `train_on_simulated_data.py`.
4. Read training results from `simulated_data/training_runs/`.

That path keeps experiments reproducible and separates data generation, quality inspection, and model training clearly.