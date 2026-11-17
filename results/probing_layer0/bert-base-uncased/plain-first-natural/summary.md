# Stronger probes: bert-base-uncased (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 0 | 83.18 | 82.55 ± 0.08 | 82.5 | [4, 3, 5] |
| 3-way | bilstm | 0 | 67.72 | 67.06 ± 0.06 | 62.8 | [3, 6, 2] |
