# Memory Comparison

| Model                  | Hardware   |   Peak VRAM (MB) |   Forward (MB) |   Backward (MB) | 8GB Target   | 10GB Target   | Reduction vs Baseline   |
|:-----------------------|:-----------|-----------------:|---------------:|----------------:|:-------------|:--------------|:------------------------|
| Baseline (ResNet-BK)   | RTX 3080   |          1902.39 |         992.2  |         1777.47 | ✅            | ✅             | nan                     |
| Phase 1 (AR-SSM + HTT) | RTX 3080   |          1810.39 |         958.02 |         1743.29 | ✅            | ✅             | 4.8%                    |