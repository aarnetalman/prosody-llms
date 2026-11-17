# Probing muse-glimmer-30b (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 8 / 52 | 82.69 | 82.19 | 82.19 | 82.13 |
| 2-way | lda | 8 / 52 | 82.69 | 82.19 | 82.19 | 82.12 |
| 3-way | ridge | 10 / 52 | 66.73 | 66.12 | 66.11 | 61.36 |
| 3-way | lda | 9 / 52 | 66.86 | 66.13 | 66.13 | 62.31 |
