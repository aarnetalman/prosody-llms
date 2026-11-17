# Probing qwen3.5-0.8b-base (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 15 / 24 | 82.41 | 81.74 | 81.74 | 81.66 |
| 2-way | lda | 15 / 24 | 82.41 | 81.74 | 81.74 | 81.66 |
| 3-way | ridge | 15 / 24 | 66.11 | 65.65 | 65.64 | 60.73 |
| 3-way | lda | 15 / 24 | 66.34 | 65.74 | 65.73 | 61.86 |
