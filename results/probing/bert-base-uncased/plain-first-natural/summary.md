# Probing bert-base-uncased (plain-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 4 / 12 | 82.48 | 81.84 | 81.84 | 81.76 |
| 2-way | lda | 4 / 12 | 82.48 | 81.84 | 81.84 | 81.76 |
| 3-way | ridge | 4 / 12 | 65.05 | 64.46 | 64.46 | 59.08 |
| 3-way | lda | 4 / 12 | 65.20 | 64.61 | 64.60 | 60.40 |
