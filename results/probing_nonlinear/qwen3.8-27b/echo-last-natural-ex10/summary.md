# Stronger probes: qwen3.8-27b (echo-last-natural-ex10)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 49 | 76.83 | 75.85 ± 0.03 | 75.8 | [10, 10, 9] |
| 3-way | bilstm | 49 | 57.78 | 56.95 ± 0.29 | 50.1 | [9, 10, 7] |
