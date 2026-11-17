# Probing hubert-large (audio-mean)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 13 / 24 | 88.01 | 88.23 | 88.23 | 88.19 |
| 2-way | lda | 13 / 24 | 88.01 | 88.23 | 88.24 | 88.19 |
| 3-way | ridge | 13 / 24 | 77.24 | 78.07 | 78.07 | 75.46 |
| 3-way | lda | 13 / 24 | 78.11 | 78.82 | 78.82 | 77.13 |
