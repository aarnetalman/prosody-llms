# Probing bert-base-cased (plain-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 4 / 12 | 82.39 | 81.83 | 81.83 | 81.74 |
| 2-way | lda | 4 / 12 | 82.39 | 81.83 | 81.83 | 81.74 |
| 3-way | ridge | 6 / 12 | 64.95 | 64.43 | 64.42 | 58.96 |
| 3-way | lda | 6 / 12 | 65.18 | 64.57 | 64.56 | 60.32 |
