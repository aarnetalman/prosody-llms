# Stronger probes: olmo-3-7b (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 21 | 82.47 | 81.85 ± 0.07 | 81.8 | [7, 1, 2] |
| 2-way | mlp | 21 | 83.26 | 82.63 ± 0.08 | 82.6 | [4, 3, 2] |
| 2-way | bilstm | 21 | 83.42 | 82.84 ± 0.06 | 82.8 | [3, 3, 1] |
| 3-way | logistic | 17 | 66.50 | 65.77 ± 0.13 | 61.1 | [3, 1, 3] |
| 3-way | mlp | 17 | 68.27 | 67.51 ± 0.09 | 63.3 | [2, 2, 2] |
| 3-way | bilstm | 17 | 68.54 | 67.84 ± 0.05 | 63.9 | [3, 3, 4] |
