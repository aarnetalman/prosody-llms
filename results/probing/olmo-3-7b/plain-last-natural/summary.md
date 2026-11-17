# Probing olmo-3-7b (plain-last-natural)

Best layer chosen on dev; test accuracy on the 2019 subset and the full file.

| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |
|---|---|---|---|---|---|---|
| 2-way | ridge | 5 / 32 | 82.34 | 81.82 | 81.82 | 81.75 |
| 2-way | lda | 5 / 32 | 82.34 | 81.82 | 81.82 | 81.75 |
| 3-way | ridge | 20 / 32 | 66.14 | 65.45 | 65.44 | 60.39 |
| 3-way | lda | 19 / 32 | 66.28 | 65.54 | 65.54 | 61.46 |
