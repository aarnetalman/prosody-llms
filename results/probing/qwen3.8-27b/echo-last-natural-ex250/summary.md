# Probing qwen3.8-27b (echo-last-natural-ex250)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 0 / 64 | 80.90 | 80.24 | 80.24 | 80.20 |
| 2-way | lda | 51 / 64 | 80.91 | 80.01 | 80.01 | 80.00 |
| 3-way | ridge | 51 / 64 | 62.21 | 61.33 | 61.32 | 53.80 |
| 3-way | lda | 51 / 64 | 63.66 | 62.70 | 62.70 | 58.19 |
