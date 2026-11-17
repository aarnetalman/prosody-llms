# Probing bert-base-uncased (plain-first-natural-n500)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 4 / 12 | 82.03 | 81.38 | 81.39 | 81.31 |
| 2-way | lda | 4 / 12 | 82.01 | 81.40 | 81.41 | 81.33 |
| 3-way | ridge | 4 / 12 | 62.79 | 62.47 | 62.47 | 56.12 |
| 3-way | lda | 8 / 12 | 63.20 | 62.74 | 62.73 | 58.03 |
