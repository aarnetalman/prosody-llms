# Stronger probes: qwen3.8-27b (echo-last-natural-ex100)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 51 | 79.65 | 78.99 ± 0.14 | 78.9 | [3, 4, 2] |
| 3-way | bilstm | 51 | 61.47 | 60.80 ± 0.07 | 55.5 | [3, 3, 3] |
