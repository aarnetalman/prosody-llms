# Probing qwen3.8-27b (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 53 / 64 | 82.99 | 82.35 | 82.35 | 82.27 |
| 2-way | lda | 53 / 64 | 82.99 | 82.35 | 82.35 | 82.27 |
| 3-way | ridge | 53 / 64 | 67.27 | 66.70 | 66.70 | 62.04 |
| 3-way | lda | 54 / 64 | 67.46 | 66.73 | 66.72 | 63.05 |
