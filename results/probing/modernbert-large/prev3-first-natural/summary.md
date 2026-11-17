# Probing modernbert-large (prev3-first-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 25 / 28 | 82.64 | 81.86 | 81.87 | 81.78 |
| 2-way | lda | 25 / 28 | 82.64 | 81.85 | 81.85 | 81.77 |
| 3-way | ridge | 19 / 28 | 65.65 | 65.16 | 65.16 | 59.96 |
| 3-way | lda | 19 / 28 | 65.87 | 65.33 | 65.33 | 61.33 |
