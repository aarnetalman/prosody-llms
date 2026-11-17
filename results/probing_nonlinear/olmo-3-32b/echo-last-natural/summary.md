# Stronger probes: olmo-3-32b (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 26 | 82.46 | 81.79 ± 0.04 | 81.7 | [5, 6, 4] |
| 2-way | mlp | 26 | 83.33 | 82.65 ± 0.05 | 82.6 | [4, 3, 2] |
| 2-way | bilstm | 26 | 83.50 | 82.76 ± 0.03 | 82.7 | [4, 3, 4] |
| 3-way | logistic | 27 | 66.66 | 66.01 ± 0.08 | 61.4 | [3, 1, 3] |
| 3-way | mlp | 27 | 68.44 | 67.67 ± 0.11 | 63.2 | [2, 2, 2] |
| 3-way | bilstm | 27 | 68.55 | 67.88 ± 0.07 | 63.8 | [1, 2, 2] |
