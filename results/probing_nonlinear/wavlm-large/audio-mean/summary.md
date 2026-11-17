# Stronger probes: wavlm-large (audio-mean)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 16 | 88.77 | 88.76 ± 0.06 | 88.7 | [8, 6, 5] |
| 2-way | mlp | 16 | 90.11 | 90.25 ± 0.09 | 90.2 | [4, 3, 4] |
| 2-way | bilstm | 16 | 90.69 | 90.64 ± 0.04 | 90.6 | [3, 3, 3] |
| 3-way | logistic | 13 | 78.70 | 79.17 ± 0.07 | 77.1 | [10, 8, 6] |
| 3-way | mlp | 13 | 82.37 | 82.76 ± 0.05 | 80.9 | [4, 3, 5] |
| 3-way | bilstm | 13 | 83.55 | 83.93 ± 0.11 | 82.3 | [4, 4, 4] |
