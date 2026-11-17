# Probing bert-base-uncased (prev3-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 5 / 12 | 82.51 | 81.80 | 81.80 | 81.71 |
| 2-way | lda | 5 / 12 | 82.51 | 81.80 | 81.80 | 81.71 |
| 3-way | ridge | 4 / 12 | 65.00 | 64.48 | 64.47 | 59.09 |
| 3-way | lda | 8 / 12 | 65.16 | 64.67 | 64.67 | 60.44 |
