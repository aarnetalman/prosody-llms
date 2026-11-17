# Probing qwen3.5-4b-base (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 4 / 32 | 82.33 | 81.70 | 81.71 | 81.63 |
| 2-way | lda | 4 / 32 | 82.33 | 81.71 | 81.71 | 81.63 |
| 3-way | ridge | 19 / 32 | 66.08 | 65.45 | 65.44 | 60.47 |
| 3-way | lda | 19 / 32 | 66.14 | 65.51 | 65.50 | 61.58 |
