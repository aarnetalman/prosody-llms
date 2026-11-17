# Probing muse-glimmer-30b (echo-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 10 / 52 | 83.21 | 82.59 | 82.59 | 82.52 |
| 2-way | lda | 10 / 52 | 83.21 | 82.59 | 82.59 | 82.52 |
| 3-way | ridge | 10 / 52 | 67.51 | 66.99 | 66.98 | 62.39 |
| 3-way | lda | 10 / 52 | 67.70 | 66.91 | 66.91 | 63.26 |
