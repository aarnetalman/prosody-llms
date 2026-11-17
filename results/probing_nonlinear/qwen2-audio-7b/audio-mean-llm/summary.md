# Stronger probes: qwen2-audio-7b (audio-mean-llm)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | logistic | 1 | 86.33 | 86.30 ± 0.05 | 86.2 | [4, 5, 5] |
| 2-way | mlp | 1 | 87.25 | 87.24 ± 0.05 | 87.2 | [4, 5, 5] |
| 2-way | bilstm | 1 | 87.36 | 87.30 ± 0.10 | 87.2 | [8, 5, 8] |
| 3-way | logistic | 3 | 73.72 | 74.06 ± 0.24 | 71.3 | [7, 8, 4] |
| 3-way | mlp | 3 | 75.68 | 76.01 ± 0.15 | 73.4 | [6, 8, 5] |
| 3-way | bilstm | 3 | 76.18 | 76.48 ± 0.14 | 73.9 | [10, 4, 7] |
