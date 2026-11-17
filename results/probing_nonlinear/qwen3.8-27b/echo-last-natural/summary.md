# Stronger probes: qwen3.8-27b (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 53 | 82.60 | 82.04 ± 0.03 | 82.0 | [3, 6, 2] |
| 2-way | mlp | 53 | 83.43 | 82.80 ± 0.03 | 82.7 | [3, 1, 2] |
| 2-way | bilstm | 53 | 83.41 | 82.89 ± 0.10 | 82.8 | [1, 2, 3] |
| 3-way | logistic | 54 | 66.71 | 66.05 ± 0.27 | 61.6 | [3, 6, 2] |
| 3-way | mlp | 54 | 68.46 | 67.82 ± 0.10 | 63.5 | [2, 2, 4] |
| 3-way | bilstm | 54 | 68.60 | 67.83 ± 0.07 | 63.7 | [4, 2, 1] |
