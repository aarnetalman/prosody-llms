# Probing qwen3.5-4b-base (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 5 / 32 | 82.74 | 82.09 | 82.09 | 82.02 |
| 2-way | lda | 5 / 32 | 82.74 | 82.09 | 82.09 | 82.02 |
| 3-way | ridge | 19 / 32 | 66.90 | 66.24 | 66.23 | 61.47 |
| 3-way | lda | 19 / 32 | 67.01 | 66.29 | 66.29 | 62.56 |
