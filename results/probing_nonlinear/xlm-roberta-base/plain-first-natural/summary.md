# Stronger probes: xlm-roberta-base (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 4 | 82.43 | 81.85 ± 0.04 | 81.8 | [4, 2, 1] |
| 2-way | mlp | 4 | 83.19 | 82.59 ± 0.05 | 82.5 | [3, 3, 2] |
| 2-way | bilstm | 4 | 83.46 | 82.87 ± 0.02 | 82.8 | [3, 2, 3] |
| 3-way | logistic | 7 | 65.70 | 65.15 ± 0.05 | 60.3 | [2, 3, 2] |
| 3-way | mlp | 7 | 67.77 | 67.17 ± 0.06 | 62.9 | [3, 5, 3] |
| 3-way | bilstm | 7 | 68.44 | 67.74 ± 0.07 | 63.7 | [3, 5, 2] |
