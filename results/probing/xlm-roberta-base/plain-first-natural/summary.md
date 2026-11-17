# Probing xlm-roberta-base (plain-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 4 / 12 | 82.47 | 81.80 | 81.80 | 81.72 |
| 2-way | lda | 4 / 12 | 82.47 | 81.82 | 81.82 | 81.74 |
| 3-way | ridge | 8 / 12 | 65.36 | 64.82 | 64.81 | 59.44 |
| 3-way | lda | 7 / 12 | 65.44 | 64.95 | 64.95 | 60.76 |
