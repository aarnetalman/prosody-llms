# Probing bert-large-uncased (plain-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 12 / 24 | 82.53 | 81.96 | 81.96 | 81.87 |
| 2-way | lda | 12 / 24 | 82.53 | 81.96 | 81.96 | 81.88 |
| 3-way | ridge | 13 / 24 | 65.46 | 64.96 | 64.96 | 59.70 |
| 3-way | lda | 13 / 24 | 65.76 | 65.04 | 65.04 | 60.94 |
