# Probing xlm-roberta-large (plain-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 9 / 24 | 82.60 | 82.05 | 82.05 | 81.96 |
| 2-way | lda | 9 / 24 | 82.60 | 82.04 | 82.04 | 81.96 |
| 3-way | ridge | 10 / 24 | 65.64 | 65.29 | 65.29 | 60.25 |
| 3-way | lda | 11 / 24 | 65.81 | 65.17 | 65.17 | 61.11 |
