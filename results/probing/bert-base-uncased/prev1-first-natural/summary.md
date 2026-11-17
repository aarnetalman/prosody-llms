# Probing bert-base-uncased (prev1-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 5 / 12 | 82.51 | 81.74 | 81.75 | 81.66 |
| 2-way | lda | 5 / 12 | 82.51 | 81.74 | 81.74 | 81.65 |
| 3-way | ridge | 4 / 12 | 64.98 | 64.46 | 64.46 | 59.07 |
| 3-way | lda | 8 / 12 | 65.17 | 64.72 | 64.71 | 60.48 |
