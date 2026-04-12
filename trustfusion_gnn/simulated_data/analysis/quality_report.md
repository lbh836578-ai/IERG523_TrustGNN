# Simulated Data Quality Report

## Overall Assessment

- Total samples: 4200
- Overall fault density: 0.0846
- Overall faulty sample ratio: 0.8060

## Split Summary

### train
- Sample count: 3000
- Fault density: 0.0847
- Faulty sample ratio: 0.8060
- Avg credibility target: 0.9153

### val
- Sample count: 600
- Fault density: 0.0824
- Faulty sample ratio: 0.7833
- Avg credibility target: 0.9176

### test
- Sample count: 600
- Fault density: 0.0860
- Faulty sample ratio: 0.8283
- Avg credibility target: 0.9140

## Fault Coverage

- BIAS: 1109
- DRIFT: 1123
- NOISE: 1125
- RANDOM: 1189
- SPIKE: 1113
- STUCK_AT: 1111

## Artifacts

- Sensor distributions: sensor_distributions.svg
- Correlation heatmap: sensor_correlation_heatmap.svg
- Fault coverage chart: fault_coverage.svg
