# Stronger probes: qwen2-audio-7b (audio-mean-encoder)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 22 | 88.32 | 88.43 ± 0.01 | 88.4 | [10, 10, 9] |
| 2-way | mlp | 22 | 90.27 | 90.50 ± 0.18 | 90.5 | [5, 8, 8] |
| 2-way | bilstm | 22 | 90.95 | 90.97 ± 0.04 | 91.0 | [5, 8, 8] |
| 3-way | logistic | 22 | 78.24 | 78.66 ± 0.02 | 76.7 | [10, 10, 10] |
| 3-way | mlp | 22 | 81.79 | 82.19 ± 0.06 | 80.6 | [10, 10, 9] |
| 3-way | bilstm | 22 | 83.84 | 84.20 ± 0.13 | 82.5 | [5, 7, 5] |
