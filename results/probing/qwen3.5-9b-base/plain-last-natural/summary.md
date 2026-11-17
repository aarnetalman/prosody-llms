# Probing qwen3.5-9b-base (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 23 / 32 | 82.47 | 81.82 | 81.81 | 81.74 |
| 2-way | lda | 23 / 32 | 82.47 | 81.82 | 81.82 | 81.75 |
| 3-way | ridge | 21 / 32 | 66.37 | 65.65 | 65.64 | 60.75 |
| 3-way | lda | 21 / 32 | 66.49 | 65.73 | 65.72 | 61.85 |
