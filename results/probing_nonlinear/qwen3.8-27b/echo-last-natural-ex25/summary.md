# Stronger probes: qwen3.8-27b (echo-last-natural-ex25)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 51 | 78.79 | 78.02 ± 0.05 | 77.9 | [5, 5, 5] |
| 3-way | bilstm | 51 | 59.83 | 58.85 ± 0.18 | 52.8 | [5, 5, 5] |
