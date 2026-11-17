# Probing qwen3.8-27b (echo-last-natural-ex10)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 51 / 64 | 77.88 | 76.97 | 76.97 | 76.94 |
| 2-way | lda | 49 / 64 | 76.88 | 76.16 | 76.16 | 76.11 |
| 3-way | ridge | 51 / 64 | 57.69 | 57.06 | 57.05 | 49.79 |
| 3-way | lda | 49 / 64 | 57.84 | 57.13 | 57.12 | 50.54 |
