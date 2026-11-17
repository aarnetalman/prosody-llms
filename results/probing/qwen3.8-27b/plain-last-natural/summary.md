# Probing qwen3.8-27b (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 52 / 64 | 82.47 | 81.90 | 81.90 | 81.82 |
| 2-way | lda | 52 / 64 | 82.47 | 81.90 | 81.90 | 81.82 |
| 3-way | ridge | 51 / 64 | 66.48 | 65.81 | 65.80 | 60.94 |
| 3-way | lda | 51 / 64 | 66.62 | 65.83 | 65.82 | 62.01 |
