# Probing qwen3.8-27b (plain-last-natural-n5000)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 54 / 64 | 82.12 | 81.56 | 81.56 | 81.48 |
| 2-way | lda | 54 / 64 | 82.20 | 81.50 | 81.50 | 81.42 |
| 3-way | ridge | 51 / 64 | 65.32 | 64.77 | 64.76 | 59.57 |
| 3-way | lda | 50 / 64 | 65.44 | 64.58 | 64.58 | 60.71 |
