# Stronger probes: muse-glimmer-30b (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 10 | 82.67 | 82.17 ± 0.03 | 82.1 | [1, 1, 3] |
| 2-way | mlp | 10 | 83.51 | 82.99 ± 0.03 | 82.9 | [2, 2, 2] |
| 2-way | bilstm | 10 | 83.43 | 82.78 ± 0.06 | 82.7 | [2, 1, 4] |
| 3-way | logistic | 10 | 66.72 | 66.14 ± 0.13 | 61.4 | [1, 3, 3] |
| 3-way | mlp | 10 | 68.48 | 67.95 ± 0.20 | 63.7 | [1, 2, 2] |
| 3-way | bilstm | 10 | 68.52 | 67.94 ± 0.03 | 63.8 | [4, 1, 4] |
