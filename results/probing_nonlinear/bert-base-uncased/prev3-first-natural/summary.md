# Stronger probes: bert-base-uncased (prev3-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 5 | 83.56 | 82.93 ± 0.03 | 82.9 | [4, 6, 3] |
| 3-way | bilstm | 8 | 68.39 | 67.69 ± 0.06 | 63.7 | [4, 3, 4] |
