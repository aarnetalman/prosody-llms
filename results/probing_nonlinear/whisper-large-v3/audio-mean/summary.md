# Stronger probes: whisper-large-v3 (audio-mean)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 23 | 88.58 | 88.57 ± 0.15 | 88.5 | [6, 10, 4] |
| 2-way | mlp | 23 | 90.73 | 90.78 ± 0.20 | 90.8 | [10, 5, 8] |
| 2-way | bilstm | 23 | 91.05 | 90.99 ± 0.19 | 91.0 | [3, 7, 4] |
| 3-way | logistic | 23 | 79.09 | 79.65 ± 0.05 | 77.9 | [10, 10, 10] |
| 3-way | mlp | 23 | 82.95 | 83.29 ± 0.07 | 81.7 | [7, 7, 10] |
| 3-way | bilstm | 23 | 84.30 | 84.66 ± 0.16 | 83.1 | [8, 7, 5] |
