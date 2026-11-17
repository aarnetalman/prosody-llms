# Stronger probes: bert-large-uncased (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 12 | 82.48 | 81.95 ± 0.05 | 81.9 | [3, 3, 5] |
| 2-way | mlp | 12 | 83.33 | 82.75 ± 0.08 | 82.7 | [2, 2, 5] |
| 2-way | bilstm | 12 | 83.55 | 82.98 ± 0.05 | 82.9 | [3, 3, 3] |
| 3-way | logistic | 13 | 65.76 | 65.06 ± 0.03 | 60.2 | [6, 5, 1] |
| 3-way | mlp | 13 | 67.72 | 67.25 ± 0.14 | 63.0 | [2, 4, 2] |
| 3-way | bilstm | 13 | 68.39 | 67.85 ± 0.07 | 63.8 | [3, 2, 4] |
