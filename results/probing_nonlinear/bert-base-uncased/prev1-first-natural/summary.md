# Stronger probes: bert-base-uncased (prev1-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 5 | 83.56 | 82.96 ± 0.04 | 82.9 | [2, 5, 5] |
| 3-way | bilstm | 8 | 68.35 | 67.73 ± 0.16 | 63.8 | [4, 2, 3] |
