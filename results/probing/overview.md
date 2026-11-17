# Probing overview

Frozen models, linear probes trained on all of `train_360`. Best layer chosen on dev (LDA probe); test accuracy on the 2019 test subset (4,821 utterances, 90,004 words) with a 95% utterance-bootstrap interval (1000 resamples). Class-1 recall: the middle class ("somewhat prominent") in the 3-way task.

## 2-way

| model | input | best layer | dev | test (2019 subset) [95% CI] | macro-F1 | ridge test | class-1 recall LDA / ridge |
|---|---|---|---|---|---|---|---|
| muse-glimmer-30b | echo | 10 / 52 | 83.21 | 82.59 [82.28, 82.88] | 82.5 | 82.59 | 85.6 / 85.6 |
| qwen3.8-27b | echo | 53 / 64 | 82.99 | 82.35 [82.03, 82.65] | 82.3 | 82.35 | 85.4 / 85.4 |
| muse-glimmer-30b | plain | 8 / 52 | 82.69 | 82.19 [81.87, 82.49] | 82.1 | 82.19 | 84.9 / 84.9 |
| olmo-3-32b | echo | 26 / 64 | 82.92 | 82.16 [81.85, 82.47] | 82.1 | 82.21 | 85.0 / 85.0 |
| qwen3.5-9b-base | echo | 19 / 32 | 82.84 | 82.12 [81.78, 82.43] | 82.0 | 82.12 | 85.2 / 85.2 |
| olmo-3-7b | echo | 21 / 32 | 82.77 | 82.11 [81.80, 82.42] | 82.0 | 82.11 | 84.9 / 84.9 |
| qwen3.5-4b-base | echo | 5 / 32 | 82.74 | 82.09 [81.76, 82.42] | 82.0 | 82.09 | 85.1 / 85.1 |
| xlm-roberta-large | plain | 9 / 24 | 82.60 | 82.04 [81.72, 82.35] | 82.0 | 82.05 | 85.7 / 85.7 |
| qwen3.5-2b-base | echo | 15 / 24 | 82.69 | 82.03 [81.71, 82.34] | 82.0 | 82.03 | 85.2 / 85.2 |
| bert-large-uncased | plain | 12 / 24 | 82.53 | 81.96 [81.63, 82.28] | 81.9 | 81.96 | 85.4 / 85.4 |
| olmo-3-32b | plain | 5 / 64 | 82.41 | 81.92 [81.58, 82.23] | 81.8 | 81.92 | 84.7 / 84.7 |
| qwen3.8-27b | plain | 52 / 64 | 82.47 | 81.90 [81.58, 82.21] | 81.8 | 81.90 | 85.1 / 85.1 |
| modernbert-large | plain | 25 / 28 | 82.67 | 81.86 [81.54, 82.18] | 81.8 | 81.86 | 85.4 / 85.4 |
| bert-base-uncased | plain | 4 / 12 | 82.48 | 81.84 [81.52, 82.16] | 81.8 | 81.84 | 85.2 / 85.2 |
| bert-base-cased | plain | 4 / 12 | 82.39 | 81.83 [81.51, 82.15] | 81.7 | 81.83 | 85.5 / 85.5 |
| olmo-3-7b | plain | 5 / 32 | 82.34 | 81.82 [81.48, 82.14] | 81.8 | 81.82 | 84.6 / 84.6 |
| qwen3.5-9b-base | plain | 23 / 32 | 82.47 | 81.82 [81.51, 82.14] | 81.8 | 81.82 | 84.8 / 84.8 |
| xlm-roberta-base | plain | 4 / 12 | 82.47 | 81.82 [81.50, 82.14] | 81.7 | 81.80 | 85.2 / 85.2 |
| qwen3.5-0.8b-base | echo | 15 / 24 | 82.41 | 81.74 [81.42, 82.06] | 81.7 | 81.74 | 85.0 / 85.0 |
| qwen3.5-4b-base | plain | 4 / 32 | 82.33 | 81.71 [81.38, 82.03] | 81.6 | 81.70 | 84.8 / 84.8 |
| qwen3.5-2b-base | plain | 2 / 24 | 82.26 | 81.70 [81.36, 82.03] | 81.6 | 81.70 | 84.8 / 84.8 |
| modernbert-base | plain | 16 / 22 | 82.42 | 81.66 [81.34, 81.97] | 81.6 | 81.65 | 85.0 / 85.0 |
| qwen3.5-0.8b-base | plain | 2 / 24 | 82.08 | 81.54 [81.21, 81.86] | 81.5 | 81.54 | 84.7 / 84.7 |

Paired differences (a − b, utterance bootstrap):

| a | b | difference [95% CI] | p |
|---|---|---|---|
| muse-glimmer-30b (echo) | muse-glimmer-30b (plain) | +0.40 [+0.28, +0.51] | 0.000 |
| olmo-3-32b (echo) | olmo-3-32b (plain) | +0.24 [+0.10, +0.39] | 0.000 |
| olmo-3-7b (echo) | olmo-3-7b (plain) | +0.29 [+0.16, +0.44] | 0.000 |
| qwen3.5-0.8b-base (echo) | qwen3.5-0.8b-base (plain) | +0.20 [+0.07, +0.33] | 0.000 |
| qwen3.5-2b-base (echo) | qwen3.5-2b-base (plain) | +0.33 [+0.20, +0.45] | 0.000 |
| qwen3.5-4b-base (echo) | qwen3.5-4b-base (plain) | +0.38 [+0.28, +0.50] | 0.000 |
| qwen3.5-9b-base (echo) | qwen3.5-9b-base (plain) | +0.30 [+0.16, +0.42] | 0.000 |
| qwen3.8-27b (echo) | qwen3.8-27b (plain) | +0.45 [+0.32, +0.57] | 0.000 |
| bert-base-cased (plain) | bert-base-uncased (plain) | -0.02 [-0.13, +0.09] | 0.756 |
| bert-large-uncased (plain) | bert-base-uncased (plain) | +0.12 [+0.00, +0.23] | 0.042 |
| modernbert-base (plain) | bert-base-uncased (plain) | -0.19 [-0.31, -0.07] | 0.000 |
| modernbert-large (plain) | bert-base-uncased (plain) | +0.02 [-0.09, +0.14] | 0.790 |
| muse-glimmer-30b (echo) | bert-base-uncased (plain) | +0.74 [+0.61, +0.88] | 0.000 |
| muse-glimmer-30b (plain) | bert-base-uncased (plain) | +0.35 [+0.21, +0.48] | 0.000 |
| olmo-3-32b (echo) | bert-base-uncased (plain) | +0.32 [+0.19, +0.47] | 0.000 |
| olmo-3-32b (plain) | bert-base-uncased (plain) | +0.08 [-0.05, +0.22] | 0.234 |
| olmo-3-7b (echo) | bert-base-uncased (plain) | +0.27 [+0.14, +0.41] | 0.000 |
| olmo-3-7b (plain) | bert-base-uncased (plain) | -0.02 [-0.16, +0.11] | 0.766 |
| qwen3.5-0.8b-base (echo) | bert-base-uncased (plain) | -0.10 [-0.24, +0.03] | 0.136 |
| qwen3.5-0.8b-base (plain) | bert-base-uncased (plain) | -0.30 [-0.43, -0.18] | 0.000 |
| qwen3.5-2b-base (echo) | bert-base-uncased (plain) | +0.18 [+0.04, +0.32] | 0.002 |
| qwen3.5-2b-base (plain) | bert-base-uncased (plain) | -0.14 [-0.27, -0.03] | 0.018 |
| qwen3.5-4b-base (echo) | bert-base-uncased (plain) | +0.25 [+0.12, +0.38] | 0.000 |
| qwen3.5-4b-base (plain) | bert-base-uncased (plain) | -0.13 [-0.26, +0.01] | 0.052 |
| qwen3.5-9b-base (echo) | bert-base-uncased (plain) | +0.28 [+0.14, +0.41] | 0.000 |
| qwen3.5-9b-base (plain) | bert-base-uncased (plain) | -0.02 [-0.15, +0.11] | 0.784 |
| qwen3.8-27b (echo) | bert-base-uncased (plain) | +0.50 [+0.36, +0.64] | 0.000 |
| qwen3.8-27b (plain) | bert-base-uncased (plain) | +0.05 [-0.08, +0.19] | 0.442 |
| xlm-roberta-base (plain) | bert-base-uncased (plain) | -0.02 [-0.14, +0.09] | 0.768 |
| xlm-roberta-large (plain) | bert-base-uncased (plain) | +0.20 [+0.08, +0.32] | 0.000 |
| muse-glimmer-30b (echo) | bert-base-cased (plain) | +0.76 [+0.63, +0.89] | 0.000 |
| muse-glimmer-30b (echo) | bert-large-uncased (plain) | +0.63 [+0.49, +0.76] | 0.000 |
| muse-glimmer-30b (echo) | modernbert-base (plain) | +0.93 [+0.79, +1.07] | 0.000 |
| muse-glimmer-30b (echo) | modernbert-large (plain) | +0.73 [+0.58, +0.86] | 0.000 |
| muse-glimmer-30b (echo) | olmo-3-32b (echo) | +0.42 [+0.29, +0.54] | 0.000 |
| muse-glimmer-30b (echo) | olmo-3-32b (plain) | +0.66 [+0.53, +0.80] | 0.000 |
| muse-glimmer-30b (echo) | olmo-3-7b (echo) | +0.48 [+0.34, +0.60] | 0.000 |
| muse-glimmer-30b (echo) | olmo-3-7b (plain) | +0.77 [+0.64, +0.89] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-0.8b-base (echo) | +0.85 [+0.71, +0.98] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-0.8b-base (plain) | +1.05 [+0.91, +1.19] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-2b-base (echo) | +0.56 [+0.44, +0.68] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-2b-base (plain) | +0.89 [+0.75, +1.02] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-4b-base (echo) | +0.50 [+0.38, +0.61] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-4b-base (plain) | +0.88 [+0.75, +1.02] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-9b-base (echo) | +0.47 [+0.35, +0.59] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-9b-base (plain) | +0.76 [+0.63, +0.88] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.8-27b (echo) | +0.24 [+0.12, +0.36] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.8-27b (plain) | +0.69 [+0.56, +0.82] | 0.000 |
| muse-glimmer-30b (echo) | xlm-roberta-base (plain) | +0.76 [+0.62, +0.90] | 0.000 |
| muse-glimmer-30b (echo) | xlm-roberta-large (plain) | +0.54 [+0.42, +0.67] | 0.000 |

## 3-way

| model | input | best layer | dev | test (2019 subset) [95% CI] | macro-F1 | ridge test | class-1 recall LDA / ridge |
|---|---|---|---|---|---|---|---|
| muse-glimmer-30b | echo | 10 / 52 | 67.70 | 66.91 [66.57, 67.24] | 63.3 | 66.99 | 50.2 / 41.8 |
| qwen3.8-27b | echo | 54 / 64 | 67.46 | 66.73 [66.36, 67.07] | 63.0 | 66.70 | 49.7 / 40.9 |
| olmo-3-32b | echo | 27 / 64 | 67.37 | 66.71 [66.35, 67.04] | 63.0 | 66.68 | 49.7 / 40.8 |
| qwen3.5-9b-base | echo | 20 / 32 | 67.24 | 66.66 [66.29, 67.01] | 63.0 | 66.51 | 49.5 / 40.5 |
| olmo-3-7b | echo | 17 / 32 | 67.23 | 66.37 [66.01, 66.71] | 62.6 | 66.30 | 49.0 / 39.7 |
| qwen3.5-4b-base | echo | 19 / 32 | 67.01 | 66.29 [65.93, 66.61] | 62.6 | 66.24 | 49.0 / 40.1 |
| qwen3.5-2b-base | echo | 15 / 24 | 66.93 | 66.22 [65.85, 66.55] | 62.4 | 66.21 | 48.7 / 40.1 |
| muse-glimmer-30b | plain | 9 / 52 | 66.86 | 66.13 [65.79, 66.45] | 62.3 | 66.12 | 49.1 / 40.9 |
| qwen3.8-27b | plain | 51 / 64 | 66.62 | 65.83 [65.47, 66.13] | 62.0 | 65.81 | 48.4 / 39.6 |
| qwen3.5-0.8b-base | echo | 15 / 24 | 66.34 | 65.74 [65.39, 66.09] | 61.9 | 65.65 | 47.9 / 39.0 |
| qwen3.5-9b-base | plain | 21 / 32 | 66.49 | 65.73 [65.37, 66.05] | 61.9 | 65.65 | 47.7 / 39.4 |
| olmo-3-7b | plain | 19 / 32 | 66.28 | 65.54 [65.19, 65.86] | 61.5 | 65.45 | 47.0 / 39.0 |
| olmo-3-32b | plain | 39 / 64 | 66.40 | 65.51 [65.17, 65.82] | 61.5 | 65.68 | 47.1 / 39.0 |
| qwen3.5-4b-base | plain | 19 / 32 | 66.14 | 65.51 [65.14, 65.82] | 61.6 | 65.45 | 47.4 / 38.6 |
| qwen3.5-2b-base | plain | 14 / 24 | 66.07 | 65.28 [64.93, 65.61] | 61.3 | 65.29 | 47.3 / 38.6 |
| modernbert-large | plain | 19 / 28 | 66.09 | 65.26 [64.90, 65.59] | 61.2 | 65.07 | 46.1 / 36.3 |
| xlm-roberta-large | plain | 11 / 24 | 65.81 | 65.17 [64.81, 65.52] | 61.1 | 65.29 | 45.6 / 37.7 |
| bert-large-uncased | plain | 13 / 24 | 65.76 | 65.04 [64.68, 65.36] | 60.9 | 64.96 | 45.7 / 36.7 |
| xlm-roberta-base | plain | 7 / 12 | 65.44 | 64.95 [64.60, 65.30] | 60.8 | 64.82 | 45.0 / 35.5 |
| modernbert-base | plain | 15 / 22 | 65.62 | 64.93 [64.56, 65.25] | 60.9 | 64.92 | 46.4 / 37.0 |
| qwen3.5-0.8b-base | plain | 16 / 24 | 65.42 | 64.72 [64.38, 65.06] | 60.6 | 64.85 | 46.0 / 37.3 |
| bert-base-uncased | plain | 4 / 12 | 65.20 | 64.61 [64.24, 64.91] | 60.4 | 64.46 | 44.8 / 35.7 |
| bert-base-cased | plain | 6 / 12 | 65.18 | 64.57 [64.21, 64.90] | 60.3 | 64.43 | 44.7 / 35.2 |

Paired differences (a − b, utterance bootstrap):

| a | b | difference [95% CI] | p |
|---|---|---|---|
| muse-glimmer-30b (echo) | muse-glimmer-30b (plain) | +0.78 [+0.59, +0.99] | 0.000 |
| olmo-3-32b (echo) | olmo-3-32b (plain) | +1.19 [+0.97, +1.41] | 0.000 |
| olmo-3-7b (echo) | olmo-3-7b (plain) | +0.83 [+0.61, +1.06] | 0.000 |
| qwen3.5-0.8b-base (echo) | qwen3.5-0.8b-base (plain) | +1.01 [+0.80, +1.22] | 0.000 |
| qwen3.5-2b-base (echo) | qwen3.5-2b-base (plain) | +0.93 [+0.71, +1.16] | 0.000 |
| qwen3.5-4b-base (echo) | qwen3.5-4b-base (plain) | +0.78 [+0.58, +1.00] | 0.000 |
| qwen3.5-9b-base (echo) | qwen3.5-9b-base (plain) | +0.93 [+0.71, +1.15] | 0.000 |
| qwen3.8-27b (echo) | qwen3.8-27b (plain) | +0.89 [+0.67, +1.13] | 0.000 |
| bert-base-cased (plain) | bert-base-uncased (plain) | -0.03 [-0.25, +0.20] | 0.798 |
| bert-large-uncased (plain) | bert-base-uncased (plain) | +0.44 [+0.22, +0.67] | 0.000 |
| modernbert-base (plain) | bert-base-uncased (plain) | +0.32 [+0.09, +0.56] | 0.004 |
| modernbert-large (plain) | bert-base-uncased (plain) | +0.65 [+0.39, +0.88] | 0.000 |
| muse-glimmer-30b (echo) | bert-base-uncased (plain) | +2.31 [+2.07, +2.56] | 0.000 |
| muse-glimmer-30b (plain) | bert-base-uncased (plain) | +1.53 [+1.28, +1.79] | 0.000 |
| olmo-3-32b (echo) | bert-base-uncased (plain) | +2.10 [+1.87, +2.35] | 0.000 |
| olmo-3-32b (plain) | bert-base-uncased (plain) | +0.91 [+0.66, +1.16] | 0.000 |
| olmo-3-7b (echo) | bert-base-uncased (plain) | +1.77 [+1.50, +2.03] | 0.000 |
| olmo-3-7b (plain) | bert-base-uncased (plain) | +0.94 [+0.69, +1.21] | 0.000 |
| qwen3.5-0.8b-base (echo) | bert-base-uncased (plain) | +1.13 [+0.89, +1.39] | 0.000 |
| qwen3.5-0.8b-base (plain) | bert-base-uncased (plain) | +0.12 [-0.14, +0.38] | 0.312 |
| qwen3.5-2b-base (echo) | bert-base-uncased (plain) | +1.61 [+1.37, +1.86] | 0.000 |
| qwen3.5-2b-base (plain) | bert-base-uncased (plain) | +0.68 [+0.44, +0.95] | 0.000 |
| qwen3.5-4b-base (echo) | bert-base-uncased (plain) | +1.69 [+1.46, +1.92] | 0.000 |
| qwen3.5-4b-base (plain) | bert-base-uncased (plain) | +0.90 [+0.65, +1.15] | 0.000 |
| qwen3.5-9b-base (echo) | bert-base-uncased (plain) | +2.05 [+1.81, +2.31] | 0.000 |
| qwen3.5-9b-base (plain) | bert-base-uncased (plain) | +1.12 [+0.87, +1.37] | 0.000 |
| qwen3.8-27b (echo) | bert-base-uncased (plain) | +2.12 [+1.88, +2.37] | 0.000 |
| qwen3.8-27b (plain) | bert-base-uncased (plain) | +1.23 [+0.96, +1.49] | 0.000 |
| xlm-roberta-base (plain) | bert-base-uncased (plain) | +0.35 [+0.11, +0.58] | 0.010 |
| xlm-roberta-large (plain) | bert-base-uncased (plain) | +0.57 [+0.34, +0.81] | 0.000 |
| muse-glimmer-30b (echo) | bert-base-cased (plain) | +2.34 [+2.10, +2.61] | 0.000 |
| muse-glimmer-30b (echo) | bert-large-uncased (plain) | +1.87 [+1.63, +2.12] | 0.000 |
| muse-glimmer-30b (echo) | modernbert-base (plain) | +1.99 [+1.73, +2.22] | 0.000 |
| muse-glimmer-30b (echo) | modernbert-large (plain) | +1.65 [+1.41, +1.91] | 0.000 |
| muse-glimmer-30b (echo) | olmo-3-32b (echo) | +0.21 [+0.01, +0.43] | 0.046 |
| muse-glimmer-30b (echo) | olmo-3-32b (plain) | +1.40 [+1.17, +1.63] | 0.000 |
| muse-glimmer-30b (echo) | olmo-3-7b (echo) | +0.54 [+0.31, +0.77] | 0.000 |
| muse-glimmer-30b (echo) | olmo-3-7b (plain) | +1.37 [+1.14, +1.60] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-0.8b-base (echo) | +1.18 [+0.96, +1.41] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-0.8b-base (plain) | +2.19 [+1.95, +2.43] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-2b-base (echo) | +0.70 [+0.48, +0.92] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-2b-base (plain) | +1.63 [+1.39, +1.86] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-4b-base (echo) | +0.62 [+0.41, +0.83] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-4b-base (plain) | +1.40 [+1.16, +1.64] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.5-9b-base (echo) | +0.25 [+0.05, +0.47] | 0.014 |
| muse-glimmer-30b (echo) | qwen3.5-9b-base (plain) | +1.19 [+0.96, +1.41] | 0.000 |
| muse-glimmer-30b (echo) | qwen3.8-27b (echo) | +0.19 [-0.04, +0.40] | 0.096 |
| muse-glimmer-30b (echo) | qwen3.8-27b (plain) | +1.08 [+0.85, +1.31] | 0.000 |
| muse-glimmer-30b (echo) | xlm-roberta-base (plain) | +1.96 [+1.71, +2.21] | 0.000 |
| muse-glimmer-30b (echo) | xlm-roberta-large (plain) | +1.74 [+1.50, +1.97] | 0.000 |

## Size series

LDA, best layer on dev; decoders with echo input. Test accuracy on the 2019 subset.

| family | model | parameters | 2-way | 3-way |
|---|---|---|---|---|
| Qwen3.5 base (+ Qwen3.8-27B) | qwen3.5-0.8b-base | 0.8B | 81.74 | 65.74 |
| Qwen3.5 base (+ Qwen3.8-27B) | qwen3.5-2b-base | 2B | 82.03 | 66.22 |
| Qwen3.5 base (+ Qwen3.8-27B) | qwen3.5-4b-base | 4B | 82.09 | 66.29 |
| Qwen3.5 base (+ Qwen3.8-27B) | qwen3.5-9b-base | 9B | 82.12 | 66.66 |
| Qwen3.5 base (+ Qwen3.8-27B) | qwen3.8-27b | 27B | 82.35 | 66.73 |
| OLMo 3 base | olmo-3-7b | 7B | 82.11 | 66.37 |
| OLMo 3 base | olmo-3-32b | 32B | 82.16 | 66.71 |
| BERT uncased | bert-base-uncased | 0.11B | 81.84 | 64.61 |
| BERT uncased | bert-large-uncased | 0.34B | 81.96 | 65.04 |
| ModernBERT | modernbert-base | 0.15B | 81.66 | 64.93 |
| ModernBERT | modernbert-large | 0.4B | 81.86 | 65.26 |
| XLM-R | xlm-roberta-base | 0.28B | 81.82 | 64.95 |
| XLM-R | xlm-roberta-large | 0.56B | 82.04 | 65.17 |

## Stronger probes

Logistic regression, MLP (one hidden layer) and BiLSTM over the frozen word vectors of the best LDA layer (per task); mean ± sd over seeds; test on the 2019 subset.

### 2-way

| model | input | layer | LDA | logistic | MLP | BiLSTM |
|---|---|---|---|---|---|---|
| bert-base-uncased | plain | 4 | 81.84 | 81.91 ± 0.03 | 82.79 ± 0.01 | 82.91 ± 0.06 |
| modernbert-base | plain | 16 | 81.66 | 81.73 ± 0.02 | 82.74 ± 0.03 | 82.90 ± 0.04 |
| muse-glimmer-30b | echo | 10 | 82.59 | 82.17 ± 0.03 | 82.99 ± 0.03 | 82.78 ± 0.06 |
| olmo-3-32b | echo | 26 | 82.16 | 81.79 ± 0.04 | 82.65 ± 0.05 | 82.76 ± 0.03 |
| olmo-3-7b | echo | 21 | 82.11 | 81.85 ± 0.07 | 82.63 ± 0.08 | 82.84 ± 0.06 |
| qwen3.8-27b | echo | 53 | 82.35 | 82.04 ± 0.03 | 82.80 ± 0.03 | 82.89 ± 0.10 |
| xlm-roberta-base | plain | 4 | 81.82 | 81.85 ± 0.04 | 82.59 ± 0.05 | 82.87 ± 0.02 |

Paired differences against frozen BERT-base + BiLSTM (seed 0):

| model | probe | difference [95% CI] | p |
|---|---|---|---|
| bert-base-uncased (plain) | mlp | -0.11 [-0.25, +0.03] | 0.128 |
| modernbert-base (plain) | bilstm | +0.03 [-0.10, +0.18] | 0.592 |
| modernbert-base (plain) | mlp | -0.19 [-0.33, -0.04] | 0.004 |
| muse-glimmer-30b (echo) | bilstm | -0.09 [-0.24, +0.06] | 0.268 |
| muse-glimmer-30b (echo) | mlp | +0.08 [-0.07, +0.24] | 0.312 |
| olmo-3-32b (echo) | bilstm | -0.18 [-0.34, -0.01] | 0.020 |
| olmo-3-32b (echo) | mlp | -0.32 [-0.49, -0.15] | 0.002 |
| olmo-3-7b (echo) | bilstm | -0.06 [-0.21, +0.10] | 0.490 |
| olmo-3-7b (echo) | mlp | -0.37 [-0.54, -0.20] | 0.000 |
| qwen3.8-27b (echo) | bilstm | -0.12 [-0.27, +0.02] | 0.112 |
| qwen3.8-27b (echo) | mlp | -0.15 [-0.30, +0.01] | 0.078 |
| xlm-roberta-base (plain) | bilstm | -0.07 [-0.22, +0.06] | 0.268 |
| xlm-roberta-base (plain) | mlp | -0.32 [-0.47, -0.17] | 0.000 |

### 3-way

| model | input | layer | LDA | logistic | MLP | BiLSTM |
|---|---|---|---|---|---|---|
| bert-base-uncased | plain | 4 | 64.61 | 64.72 ± 0.13 | 66.90 ± 0.09 | 67.72 ± 0.15 |
| modernbert-base | plain | 15 | 64.93 | 65.16 ± 0.08 | 67.21 ± 0.01 | 67.67 ± 0.11 |
| muse-glimmer-30b | echo | 10 | 66.91 | 66.14 ± 0.13 | 67.95 ± 0.20 | 67.94 ± 0.03 |
| olmo-3-32b | echo | 27 | 66.71 | 66.01 ± 0.08 | 67.67 ± 0.11 | 67.88 ± 0.07 |
| olmo-3-7b | echo | 17 | 66.37 | 65.77 ± 0.13 | 67.51 ± 0.09 | 67.84 ± 0.05 |
| qwen3.8-27b | echo | 54 | 66.73 | 66.05 ± 0.27 | 67.82 ± 0.10 | 67.83 ± 0.07 |
| xlm-roberta-base | plain | 7 | 64.95 | 65.15 ± 0.05 | 67.17 ± 0.06 | 67.74 ± 0.07 |

Paired differences against frozen BERT-base + BiLSTM (seed 0):

| model | probe | difference [95% CI] | p |
|---|---|---|---|
| bert-base-uncased (plain) | mlp | -0.79 [-1.04, -0.56] | 0.000 |
| modernbert-base (plain) | bilstm | +0.01 [-0.23, +0.24] | 0.934 |
| modernbert-base (plain) | mlp | -0.48 [-0.75, -0.23] | 0.000 |
| muse-glimmer-30b (echo) | bilstm | +0.20 [-0.06, +0.44] | 0.114 |
| muse-glimmer-30b (echo) | mlp | +0.02 [-0.24, +0.27] | 0.910 |
| olmo-3-32b (echo) | bilstm | +0.12 [-0.17, +0.39] | 0.402 |
| olmo-3-32b (echo) | mlp | +0.07 [-0.19, +0.32] | 0.638 |
| olmo-3-7b (echo) | bilstm | +0.09 [-0.17, +0.34] | 0.506 |
| olmo-3-7b (echo) | mlp | -0.24 [-0.50, +0.03] | 0.074 |
| qwen3.8-27b (echo) | bilstm | +0.07 [-0.19, +0.31] | 0.558 |
| qwen3.8-27b (echo) | mlp | +0.22 [-0.06, +0.46] | 0.108 |
| xlm-roberta-base (plain) | bilstm | +0.05 [-0.19, +0.28] | 0.678 |
| xlm-roberta-base (plain) | mlp | -0.58 [-0.84, -0.32] | 0.000 |

## Reference rows

| system | task | test (2019 subset) |
|---|---|---|
| fine-tuned BERT-base, 5 seeds | 2-way | 83.02 ± 0.13 |
| fine-tuned BERT-base, 5 seeds | 3-way | 68.49 ± 0.10 |
| word majority 2019 baseline | 2-way | 80.21 |
| word majority 2019 baseline | 3-way | 62.34 |
| majority baseline | 2-way | 51.98 |
| majority baseline | 3-way | 48.02 |
