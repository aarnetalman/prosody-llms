# Stronger probes: qwen3.8-27b (echo-last-natural-ex500)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 51 | 81.05 | 80.34 ± 0.18 | 80.3 | [3, 1, 2] |
| 3-way | bilstm | 51 | 64.17 | 63.20 ± 0.41 | 57.8 | [2, 2, 2] |
