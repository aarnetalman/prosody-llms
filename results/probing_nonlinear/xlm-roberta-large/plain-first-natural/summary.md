# Stronger probes: xlm-roberta-large (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 9 | 82.56 | 82.00 ± 0.02 | 81.9 | [3, 4, 1] |
| 2-way | mlp | 9 | 83.36 | 82.86 ± 0.03 | 82.8 | [6, 5, 4] |
| 2-way | bilstm | 9 | 83.47 | 82.86 ± 0.05 | 82.8 | [2, 5, 2] |
| 3-way | logistic | 11 | 65.87 | 65.27 ± 0.07 | 60.5 | [2, 2, 1] |
| 3-way | mlp | 11 | 67.87 | 67.26 ± 0.15 | 63.0 | [3, 3, 5] |
| 3-way | bilstm | 11 | 68.44 | 67.89 ± 0.09 | 63.8 | [5, 2, 2] |
