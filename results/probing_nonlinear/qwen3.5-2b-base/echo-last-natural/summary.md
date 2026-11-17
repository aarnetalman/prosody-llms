# Stronger probes: qwen3.5-2b-base (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 15 | 82.55 | 82.00 ± 0.04 | 81.9 | [1, 1, 2] |
| 2-way | mlp | 15 | 83.30 | 82.68 ± 0.04 | 82.6 | [4, 1, 4] |
| 2-way | bilstm | 15 | 83.49 | 82.87 ± 0.03 | 82.8 | [4, 5, 2] |
| 3-way | logistic | 15 | 66.61 | 65.89 ± 0.03 | 61.3 | [3, 1, 1] |
| 3-way | mlp | 15 | 68.35 | 67.61 ± 0.15 | 63.5 | [3, 4, 4] |
| 3-way | bilstm | 15 | 68.60 | 67.92 ± 0.07 | 63.8 | [2, 5, 4] |
