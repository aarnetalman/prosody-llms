# Stronger probes: modernbert-large (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 25 | 82.59 | 81.77 ± 0.01 | 81.7 | [3, 1, 4] |
| 2-way | mlp | 25 | 83.38 | 82.66 ± 0.06 | 82.6 | [3, 5, 4] |
| 2-way | bilstm | 25 | 83.69 | 83.03 ± 0.03 | 83.0 | [4, 5, 3] |
| 3-way | logistic | 19 | 66.06 | 65.29 ± 0.16 | 60.6 | [4, 5, 4] |
| 3-way | mlp | 19 | 68.10 | 67.26 ± 0.02 | 62.9 | [2, 4, 4] |
| 3-way | bilstm | 19 | 68.57 | 68.00 ± 0.06 | 63.8 | [2, 7, 4] |
