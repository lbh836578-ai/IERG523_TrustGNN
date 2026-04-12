# TrustFusion-GNN Optimization Plan

## Goal

Optimize the TrustFusion-GNN system incrementally while keeping the current training and inference workflow stable.

## Constraints

1. Prefer additive changes over refactoring existing core logic.
2. Keep the current training, inference, and model APIs unchanged unless a later phase proves that a change is necessary.
3. Store newly generated synthetic datasets inside `simulated_data/` for reproducibility and later experiments.

## Phase 1: Expand and Quantify Simulated Data

### Objective

Generate substantially more synthetic data than the current demo-scale dataset and attach measurable dataset statistics.

### Deliverables

1. A dedicated dataset generation script.
2. A `simulated_data/` directory containing serialized dataset splits.
3. Quantitative summaries for each split and an overall manifest.

### Success Criteria

1. The repository can generate train/validation/test synthetic splits on demand.
2. The generated files include both raw arrays and summary metadata.
3. The summaries quantify sample count, fault density, fault-type coverage, sensor-level statistics, and target-output statistics.

## Phase 2: Data Fidelity Review

### Objective

Evaluate whether the synthetic data distribution is realistic enough for model training.

### Candidate Checks

1. Compare per-sensor mean, standard deviation, and min/max ranges against expected agricultural operating ranges.
2. Inspect fault-type balance and duration balance.
3. Inspect correlations between temperature, humidity, soil moisture, and light.

## Phase 3: Training Pipeline Upgrade

### Objective

Use the larger synthetic dataset to improve model stability and training repeatability.

### Candidate Actions

1. Add dataset split loading utilities.
2. Support training directly from pre-generated files.
3. Add experiment metadata capture for reproducibility.

## Phase 4: Model Evaluation Hardening

### Objective

Make evaluation more reliable and comparable across runs.

### Candidate Actions

1. Add fixed validation and test splits.
2. Persist evaluation summaries after each run.
3. Track robustness under controlled anomaly ratios.

## Immediate Step To Execute Now

Implement Phase 1 only:

1. Generate more synthetic data.
2. Quantify the generated data.
3. Save all outputs under `simulated_data/`.
4. Avoid unnecessary changes to the rest of the system.