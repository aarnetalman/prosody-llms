# Stronger probes: hubert-large (audio-mean)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 13 | 88.83 | 88.95 ± 0.08 | 88.9 | [4, 6, 10] |
| 2-way | mlp | 13 | 90.51 | 90.64 ± 0.03 | 90.6 | [3, 3, 3] |
| 2-way | bilstm | 13 | 90.87 | 90.88 ± 0.12 | 90.9 | [3, 5, 3] |
| 3-way | logistic | 13 | 79.54 | 80.02 ± 0.07 | 78.0 | [10, 10, 10] |
| 3-way | mlp | 13 | 82.83 | 83.28 ± 0.02 | 81.6 | [4, 4, 3] |
| 3-way | bilstm | 13 | 83.73 | 84.10 ± 0.12 | 82.6 | [4, 5, 4] |
