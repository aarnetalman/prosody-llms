# Probing qwen3.5-0.8b-base (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 2 / 24 | 82.09 | 81.54 | 81.54 | 81.46 |
| 2-way | lda | 2 / 24 | 82.08 | 81.54 | 81.54 | 81.46 |
| 3-way | ridge | 15 / 24 | 65.16 | 64.85 | 64.84 | 59.64 |
| 3-way | lda | 16 / 24 | 65.42 | 64.72 | 64.71 | 60.56 |
