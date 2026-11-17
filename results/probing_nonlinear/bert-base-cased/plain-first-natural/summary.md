# Stronger probes: bert-base-cased (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 4 | 82.40 | 81.88 ± 0.03 | 81.8 | [1, 1, 4] |
| 2-way | mlp | 4 | 83.37 | 82.81 ± 0.08 | 82.7 | [6, 6, 4] |
| 2-way | bilstm | 4 | 83.52 | 82.96 ± 0.05 | 82.9 | [4, 3, 3] |
| 3-way | logistic | 6 | 65.30 | 64.71 ± 0.12 | 59.8 | [2, 3, 5] |
| 3-way | mlp | 6 | 67.62 | 67.13 ± 0.02 | 62.7 | [3, 3, 4] |
| 3-way | bilstm | 6 | 68.44 | 67.84 ± 0.08 | 63.8 | [4, 6, 3] |
