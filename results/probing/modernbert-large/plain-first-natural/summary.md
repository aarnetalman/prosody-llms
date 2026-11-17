# Probing modernbert-large (plain-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 25 / 28 | 82.67 | 81.86 | 81.86 | 81.78 |
| 2-way | lda | 25 / 28 | 82.67 | 81.86 | 81.86 | 81.78 |
| 3-way | ridge | 19 / 28 | 65.74 | 65.07 | 65.07 | 59.81 |
| 3-way | lda | 19 / 28 | 66.09 | 65.26 | 65.26 | 61.20 |
