# Stronger probes: bert-base-uncased (plain-first-natural-train_100)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 4 | 82.53 | 81.80 ± 0.04 | 81.7 | [1, 6, 4] |
| 2-way | mlp | 4 | 83.07 | 82.45 ± 0.05 | 82.4 | [3, 5, 4] |
| 2-way | bilstm | 4 | 83.22 | 82.71 ± 0.03 | 82.6 | [2, 3, 3] |
| 3-way | logistic | 4 | 65.18 | 64.59 ± 0.16 | 59.8 | [7, 3, 4] |
| 3-way | mlp | 4 | 66.69 | 66.07 ± 0.12 | 61.7 | [2, 4, 4] |
| 3-way | bilstm | 4 | 67.66 | 67.08 ± 0.10 | 62.8 | [3, 3, 3] |
