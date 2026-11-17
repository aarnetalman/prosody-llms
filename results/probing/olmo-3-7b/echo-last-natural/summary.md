# Probing olmo-3-7b (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 21 / 32 | 82.77 | 82.11 | 82.11 | 82.05 |
| 2-way | lda | 21 / 32 | 82.77 | 82.11 | 82.11 | 82.04 |
| 3-way | ridge | 17 / 32 | 67.06 | 66.30 | 66.29 | 61.46 |
| 3-way | lda | 17 / 32 | 67.23 | 66.37 | 66.37 | 62.60 |
