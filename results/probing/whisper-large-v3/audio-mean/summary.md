# Probing whisper-large-v3 (audio-mean)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 23 / 32 | 88.21 | 88.22 | 88.23 | 88.18 |
| 2-way | lda | 23 / 32 | 88.21 | 88.22 | 88.23 | 88.18 |
| 3-way | ridge | 23 / 32 | 77.48 | 78.03 | 78.03 | 75.46 |
| 3-way | lda | 23 / 32 | 78.31 | 78.74 | 78.73 | 77.07 |
