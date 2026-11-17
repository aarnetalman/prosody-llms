# Stronger probes: qwen3.8-27b (echo-last-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 0 | 82.88 | 82.28 ± 0.01 | 82.2 | [1, 2, 2] |
| 3-way | bilstm | 0 | 66.98 | 66.54 ± 0.15 | 62.1 | [3, 3, 2] |
