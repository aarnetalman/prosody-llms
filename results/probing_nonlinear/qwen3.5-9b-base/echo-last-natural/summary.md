# Stronger probes: qwen3.5-9b-base (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 19 | 82.47 | 81.80 ± 0.06 | 81.7 | [2, 1, 2] |
| 2-way | mlp | 19 | 83.30 | 82.71 ± 0.03 | 82.6 | [4, 4, 2] |
| 2-way | bilstm | 19 | 83.40 | 82.80 ± 0.06 | 82.7 | [5, 3, 6] |
| 3-way | logistic | 20 | 66.66 | 66.12 ± 0.12 | 61.7 | [3, 4, 2] |
| 3-way | mlp | 20 | 68.37 | 67.72 ± 0.12 | 63.6 | [3, 2, 2] |
| 3-way | bilstm | 20 | 68.45 | 67.83 ± 0.08 | 63.8 | [1, 2, 2] |
