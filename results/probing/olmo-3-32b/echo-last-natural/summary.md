# Probing olmo-3-32b (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 27 / 64 | 82.92 | 82.21 | 82.21 | 82.14 |
| 2-way | lda | 26 / 64 | 82.92 | 82.16 | 82.17 | 82.10 |
| 3-way | ridge | 27 / 64 | 67.19 | 66.68 | 66.67 | 61.97 |
| 3-way | lda | 27 / 64 | 67.37 | 66.71 | 66.70 | 63.00 |
