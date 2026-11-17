# Probing bert-base-uncased (plain-first-natural-n5000)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 4 / 12 | 82.40 | 81.73 | 81.74 | 81.65 |
| 2-way | lda | 4 / 12 | 82.40 | 81.73 | 81.73 | 81.65 |
| 3-way | ridge | 6 / 12 | 64.48 | 64.06 | 64.05 | 58.56 |
| 3-way | lda | 8 / 12 | 64.75 | 64.39 | 64.38 | 60.13 |
