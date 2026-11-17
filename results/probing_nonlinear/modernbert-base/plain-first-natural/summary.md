# Stronger probes: modernbert-base (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 16 | 82.41 | 81.73 ± 0.02 | 81.7 | [3, 9, 3] |
| 2-way | mlp | 16 | 83.31 | 82.74 ± 0.03 | 82.7 | [7, 9, 10] |
| 2-way | bilstm | 16 | 83.60 | 82.90 ± 0.04 | 82.8 | [4, 6, 5] |
| 3-way | logistic | 15 | 65.72 | 65.16 ± 0.08 | 60.5 | [7, 2, 5] |
| 3-way | mlp | 15 | 67.77 | 67.21 ± 0.01 | 62.9 | [4, 4, 5] |
| 3-way | bilstm | 15 | 68.31 | 67.67 ± 0.11 | 63.6 | [4, 5, 4] |
