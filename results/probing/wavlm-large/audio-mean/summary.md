# Probing wavlm-large (audio-mean)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 16 / 24 | 87.90 | 87.82 | 87.83 | 87.77 |
| 2-way | lda | 16 / 24 | 87.90 | 87.82 | 87.83 | 87.77 |
| 3-way | ridge | 10 / 24 | 76.77 | 77.42 | 77.42 | 74.64 |
| 3-way | lda | 13 / 24 | 77.56 | 77.99 | 77.99 | 76.18 |
