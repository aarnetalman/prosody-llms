# Probing qwen2-audio-7b (audio-mean-encoder)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 22 / 32 | 88.04 | 88.09 | 88.09 | 88.04 |
| 2-way | lda | 22 / 32 | 88.04 | 88.08 | 88.09 | 88.04 |
| 3-way | ridge | 23 / 32 | 76.82 | 77.26 | 77.25 | 74.62 |
| 3-way | lda | 22 / 32 | 77.86 | 78.11 | 78.11 | 76.35 |
