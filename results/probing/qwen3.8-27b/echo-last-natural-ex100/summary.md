# Probing qwen3.8-27b (echo-last-natural-ex100)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 0 / 64 | 80.54 | 79.94 | 79.93 | 79.88 |
| 2-way | lda | 51 / 64 | 79.76 | 79.03 | 79.02 | 79.01 |
| 3-way | ridge | 51 / 64 | 60.74 | 59.76 | 59.75 | 51.82 |
| 3-way | lda | 51 / 64 | 61.74 | 60.73 | 60.73 | 55.60 |
