# Probing bert-base-uncased (plain-first-natural-train_100)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 4 / 12 | 82.51 | 81.84 | 81.84 | 81.75 |
| 2-way | lda | 4 / 12 | 82.51 | 81.81 | 81.81 | 81.73 |
| 3-way | ridge | 4 / 12 | 64.87 | 64.36 | 64.35 | 58.90 |
| 3-way | lda | 4 / 12 | 65.11 | 64.42 | 64.42 | 60.21 |
