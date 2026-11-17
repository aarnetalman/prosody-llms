# Probing modernbert-base (plain-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 16 / 22 | 82.42 | 81.65 | 81.66 | 81.57 |
| 2-way | lda | 16 / 22 | 82.42 | 81.66 | 81.66 | 81.57 |
| 3-way | ridge | 17 / 22 | 65.45 | 64.92 | 64.91 | 59.76 |
| 3-way | lda | 15 / 22 | 65.62 | 64.93 | 64.92 | 60.85 |
