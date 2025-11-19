# Scaling Comparison

| Model                  | Complexity   |   R² Score |   Scaling Coefficient |   Avg Throughput (tokens/s) |
|:-----------------------|:-------------|-----------:|----------------------:|----------------------------:|
| Baseline (ResNet-BK)   | O(N)         |     0.9995 |              2.44814  |                      798.28 |
| Phase 1 (AR-SSM + HTT) | O(N log N)   |     1      |              0.290234 |                      824.74 |