# Probing muse-glimmer-30b (echo-last-natural-train_100)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 10 / 52 | 83.15 | 82.51 | 82.51 | 82.43 |
| 2-way | lda | 10 / 52 | 83.15 | 82.50 | 82.50 | 82.43 |
| 3-way | ridge | 11 / 52 | 67.15 | 66.57 | 66.56 | 61.88 |
| 3-way | lda | 10 / 52 | 67.45 | 66.73 | 66.73 | 63.14 |
