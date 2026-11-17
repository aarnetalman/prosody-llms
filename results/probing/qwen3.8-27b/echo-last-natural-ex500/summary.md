# Probing qwen3.8-27b (echo-last-natural-ex500)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 1 / 64 | 81.24 | 80.63 | 80.63 | 80.60 |
| 2-way | lda | 51 / 64 | 81.27 | 80.70 | 80.70 | 80.67 |
| 3-way | ridge | 51 / 64 | 62.97 | 62.34 | 62.34 | 56.34 |
| 3-way | lda | 51 / 64 | 64.01 | 63.01 | 63.00 | 58.63 |
