# Stronger probes: random-words-1024 (plain-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 0 | 83.09 | 82.49 ± 0.05 | 82.4 | [2, 2, 2] |
| 3-way | bilstm | 0 | 67.23 | 66.59 ± 0.12 | 62.3 | [3, 4, 4] |
