# Probing qwen2-audio-7b (audio-mean-llm)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 1 / 32 | 86.63 | 86.50 | 86.50 | 86.43 |
| 2-way | lda | 1 / 32 | 86.56 | 86.45 | 86.45 | 86.37 |
| 3-way | ridge | 2 / 32 | 73.58 | 73.77 | 73.78 | 70.32 |
| 3-way | lda | 3 / 32 | 74.19 | 74.39 | 74.39 | 72.11 |
