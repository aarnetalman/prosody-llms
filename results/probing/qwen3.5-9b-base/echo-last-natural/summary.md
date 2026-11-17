# Probing qwen3.5-9b-base (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 19 / 32 | 82.84 | 82.12 | 82.13 | 82.05 |
| 2-way | lda | 19 / 32 | 82.84 | 82.12 | 82.12 | 82.04 |
| 3-way | ridge | 20 / 32 | 67.03 | 66.51 | 66.51 | 61.81 |
| 3-way | lda | 20 / 32 | 67.24 | 66.66 | 66.66 | 63.00 |
