# Probing olmo-3-32b (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 5 / 64 | 82.42 | 81.92 | 81.92 | 81.86 |
| 2-way | lda | 5 / 64 | 82.41 | 81.92 | 81.92 | 81.85 |
| 3-way | ridge | 27 / 64 | 66.29 | 65.68 | 65.68 | 60.66 |
| 3-way | lda | 39 / 64 | 66.40 | 65.51 | 65.51 | 61.53 |
