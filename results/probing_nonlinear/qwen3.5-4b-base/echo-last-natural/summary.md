# Stronger probes: qwen3.5-4b-base (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 5 | 82.51 | 81.94 ± 0.02 | 81.9 | [2, 1, 4] |
| 2-way | mlp | 5 | 83.21 | 82.60 ± 0.01 | 82.5 | [2, 1, 2] |
| 2-way | bilstm | 5 | 83.52 | 82.90 ± 0.02 | 82.8 | [3, 2, 5] |
| 3-way | logistic | 19 | 66.61 | 65.99 ± 0.12 | 61.4 | [1, 1, 3] |
| 3-way | mlp | 19 | 68.33 | 67.64 ± 0.11 | 63.6 | [3, 2, 3] |
| 3-way | bilstm | 19 | 68.60 | 67.85 ± 0.18 | 63.7 | [3, 6, 2] |
