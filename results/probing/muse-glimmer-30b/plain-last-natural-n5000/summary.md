# Probing muse-glimmer-30b (plain-last-natural-n5000)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 6 / 52 | 82.29 | 81.85 | 81.85 | 81.79 |
| 2-way | lda | 6 / 52 | 82.33 | 81.91 | 81.91 | 81.85 |
| 3-way | ridge | 37 / 52 | 65.48 | 64.67 | 64.66 | 59.41 |
| 3-way | lda | 34 / 52 | 65.69 | 64.92 | 64.91 | 61.00 |
