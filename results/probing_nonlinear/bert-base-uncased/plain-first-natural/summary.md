# Stronger probes: bert-base-uncased (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 4 | 82.56 | 81.91 ± 0.03 | 81.8 | [2, 1, 3] |
| 2-way | mlp | 4 | 83.40 | 82.79 ± 0.01 | 82.7 | [6, 6, 4] |
| 2-way | bilstm | 4 | 83.54 | 82.91 ± 0.06 | 82.8 | [3, 4, 3] |
| 3-way | logistic | 4 | 65.21 | 64.72 ± 0.13 | 59.8 | [4, 6, 1] |
| 3-way | mlp | 4 | 67.57 | 66.90 ± 0.09 | 62.5 | [3, 5, 7] |
| 3-way | bilstm | 4 | 68.35 | 67.72 ± 0.15 | 63.6 | [6, 6, 3] |
