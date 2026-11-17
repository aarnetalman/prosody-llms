# Stronger probes: qwen3.5-0.8b-base (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 15 | 82.36 | 81.79 ± 0.04 | 81.7 | [2, 1, 5] |
| 2-way | mlp | 15 | 83.29 | 82.69 ± 0.09 | 82.6 | [4, 4, 5] |
| 2-way | bilstm | 15 | 83.56 | 82.92 ± 0.03 | 82.8 | [2, 2, 2] |
| 3-way | logistic | 15 | 66.22 | 65.69 ± 0.10 | 61.0 | [3, 1, 1] |
| 3-way | mlp | 15 | 68.25 | 67.57 ± 0.06 | 63.3 | [4, 5, 5] |
| 3-way | bilstm | 15 | 68.72 | 67.98 ± 0.10 | 64.0 | [3, 5, 4] |
