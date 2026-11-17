# Probing qwen3.8-27b (echo-last-natural-ex25)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 1 / 64 | 80.07 | 79.58 | 79.58 | 79.54 |
| 2-way | lda | 51 / 64 | 79.03 | 78.19 | 78.19 | 78.11 |
| 3-way | ridge | 51 / 64 | 59.94 | 59.18 | 59.17 | 52.08 |
| 3-way | lda | 51 / 64 | 59.67 | 58.79 | 58.78 | 53.52 |
