# Probing qwen3.5-2b-base (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 2 / 24 | 82.26 | 81.70 | 81.70 | 81.62 |
| 2-way | lda | 2 / 24 | 82.26 | 81.70 | 81.70 | 81.62 |
| 3-way | ridge | 15 / 24 | 65.95 | 65.29 | 65.28 | 60.25 |
| 3-way | lda | 14 / 24 | 66.07 | 65.28 | 65.27 | 61.27 |
