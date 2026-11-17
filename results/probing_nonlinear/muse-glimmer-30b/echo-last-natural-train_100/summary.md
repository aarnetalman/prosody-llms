# Stronger probes: muse-glimmer-30b (echo-last-natural-train_100)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 10 | 82.63 | 82.06 ± 0.05 | 82.0 | [1, 2, 6] |
| 2-way | mlp | 10 | 83.24 | 82.63 ± 0.12 | 82.5 | [2, 2, 1] |
| 2-way | bilstm | 10 | 83.31 | 82.68 ± 0.05 | 82.6 | [2, 1, 3] |
| 3-way | logistic | 10 | 66.52 | 66.00 ± 0.24 | 61.6 | [2, 3, 4] |
| 3-way | mlp | 10 | 67.78 | 67.36 ± 0.05 | 63.2 | [2, 1, 2] |
| 3-way | bilstm | 10 | 68.09 | 67.53 ± 0.08 | 63.5 | [2, 1, 3] |
