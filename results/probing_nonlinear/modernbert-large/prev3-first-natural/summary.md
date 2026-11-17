# Stronger probes: modernbert-large (prev3-first-natural)

Seeds [0, 1, 2]; mean ± sd over seeds; test on the 2019 subset.

| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |
|---|---|---|---|---|---|---|
| 2-way | bilstm | 25 | 83.70 | 82.95 ± 0.06 | 82.9 | [2, 5, 4] |
| 3-way | bilstm | 19 | 68.52 | 67.88 ± 0.09 | 63.8 | [6, 4, 2] |
