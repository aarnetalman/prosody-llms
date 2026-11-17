# Probing qwen3.5-2b-base (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 15 / 24 | 82.69 | 82.03 | 82.03 | 81.95 |
| 2-way | lda | 15 / 24 | 82.69 | 82.03 | 82.03 | 81.95 |
| 3-way | ridge | 15 / 24 | 66.80 | 66.21 | 66.20 | 61.43 |
| 3-way | lda | 15 / 24 | 66.93 | 66.22 | 66.21 | 62.44 |
