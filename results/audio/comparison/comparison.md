# Text against audio on identical words

Common words: 89935 of 90063 scored test words (aligned, starting within 30 s). 95% intervals: utterance-level bootstrap. McNemar: against the best text system on the common words.

## 3-way

| system | modality | family | common words | 95% CI | all words | vs best text | p |
|---|---|---|---|---|---|---|---|
| wavelet toolkit (labelling procedure) | audio | reference | 91.50 | 91.30-91.70 | 91.43 | +23.43 | 0 |
| whisper-large-v3 (audio-mean) bilstm | audio | probe_nonlinear | 84.60 | 84.32-84.86 | 84.53 | +16.53 | 0 |
| hubert-large (audio-mean) bilstm | audio | probe_nonlinear | 84.19 | 83.91-84.47 | 84.14 | +16.12 | 0 |
| qwen2-audio-7b (audio-mean-encoder) bilstm | audio | probe_nonlinear | 84.12 | 83.86-84.40 | 84.06 | +16.05 | 0 |
| wavlm-large (audio-mean) bilstm | audio | probe_nonlinear | 84.04 | 83.74-84.31 | 83.99 | +15.97 | 0 |
| hubert-large (audio-mean) mlp | audio | probe_nonlinear | 83.36 | 83.06-83.65 | 83.31 | +15.29 | 0 |
| whisper-large-v3 (audio-mean) mlp | audio | probe_nonlinear | 83.34 | 83.04-83.60 | 83.27 | +15.27 | 0 |
| wavlm-large (audio-mean) mlp | audio | probe_nonlinear | 82.75 | 82.45-83.03 | 82.70 | +14.68 | 0 |
| qwen2-audio-7b (audio-mean-encoder) mlp | audio | probe_nonlinear | 82.18 | 81.90-82.46 | 82.12 | +14.11 | 0 |
| hubert-large (audio-mean) logistic | audio | probe_nonlinear | 80.01 | 79.68-80.31 | 79.96 | +11.94 | 0 |
| whisper-large-v3 (audio-mean) logistic | audio | probe_nonlinear | 79.75 | 79.46-80.00 | 79.69 | +11.68 | 0 |
| wavlm-large (audio-mean) logistic | audio | probe_nonlinear | 79.29 | 78.96-79.56 | 79.24 | +11.22 | 0 |
| hubert-large (audio-mean) LDA | audio | probe_linear | 78.86 | 78.53-79.15 | 78.82 | +10.79 | 0 |
| whisper-large-v3 (audio-mean) LDA | audio | probe_linear | 78.79 | 78.48-79.07 | 78.73 | +10.72 | 0 |
| qwen2-audio-7b (audio-mean-encoder) logistic | audio | probe_nonlinear | 78.73 | 78.39-79.03 | 78.67 | +10.66 | 0 |
| qwen2-audio-7b (audio-mean-encoder) LDA | audio | probe_linear | 78.17 | 77.84-78.46 | 78.11 | +10.10 | 0 |
| wavlm-large (audio-mean) LDA | audio | probe_linear | 78.04 | 77.70-78.36 | 77.99 | +9.97 | 0 |
| qwen2-audio-7b (audio-mean-llm) bilstm | audio | probe_nonlinear | 76.69 | 76.37-76.99 | 76.63 | +8.62 | 0 |
| qwen2-audio-7b (audio-mean-llm) mlp | audio | probe_nonlinear | 76.04 | 75.70-76.33 | 75.99 | +7.97 | 0 |
| qwen2-audio-7b (audio-mean-llm) LDA | audio | probe_linear | 74.44 | 74.12-74.75 | 74.39 | +6.37 | 0 |
| qwen2-audio-7b (audio-mean-llm) logistic | audio | probe_nonlinear | 74.19 | 73.88-74.49 | 74.14 | +6.12 | 4.9e-324 |
| modernbert-large (plain-first-natural) bilstm | text | probe_nonlinear | 68.07 | 67.69-68.37 | 68.05 |  |  |
| qwen3.5-0.8b-base (echo-last-natural) bilstm | text | probe_nonlinear | 68.07 | 67.73-68.40 | 68.05 | +0.00 | 0.99 |
| xlm-roberta-large (plain-first-natural) bilstm | text | probe_nonlinear | 67.95 | 67.56-68.29 | 67.94 | -0.12 | 0.32 |
| qwen3.8-27b (echo-last-natural) mlp | text | probe_nonlinear | 67.93 | 67.58-68.26 | 67.91 | -0.14 | 0.25 |
| bert-large-uncased (plain-first-natural) bilstm | text | probe_nonlinear | 67.92 | 67.55-68.24 | 67.91 | -0.15 | 0.21 |
| muse-glimmer-30b (echo-last-natural) bilstm | text | probe_nonlinear | 67.91 | 67.54-68.22 | 67.90 | -0.16 | 0.2 |
| qwen3.5-2b-base (echo-last-natural) bilstm | text | probe_nonlinear | 67.85 | 67.48-68.18 | 67.83 | -0.22 | 0.056 |
| olmo-3-32b (echo-last-natural) bilstm | text | probe_nonlinear | 67.83 | 67.45-68.17 | 67.81 | -0.24 | 0.047 |
| qwen3.5-9b-base (echo-last-natural) bilstm | text | probe_nonlinear | 67.81 | 67.44-68.15 | 67.79 | -0.26 | 0.031 |
| olmo-3-7b (echo-last-natural) bilstm | text | probe_nonlinear | 67.80 | 67.43-68.11 | 67.78 | -0.27 | 0.028 |
| olmo-3-32b (echo-last-natural) mlp | text | probe_nonlinear | 67.78 | 67.41-68.11 | 67.76 | -0.29 | 0.02 |
| qwen3.8-27b (echo-last-natural) bilstm | text | probe_nonlinear | 67.78 | 67.41-68.11 | 67.76 | -0.29 | 0.016 |
| bert-base-cased (plain-first-natural) bilstm | text | probe_nonlinear | 67.77 | 67.41-68.09 | 67.75 | -0.30 | 0.013 |
| xlm-roberta-base (plain-first-natural) bilstm | text | probe_nonlinear | 67.75 | 67.39-68.08 | 67.74 | -0.32 | 0.0075 |
| muse-glimmer-30b (echo-last-natural) mlp | text | probe_nonlinear | 67.73 | 67.37-68.03 | 67.72 | -0.34 | 0.0056 |
| qwen3.5-9b-base (echo-last-natural) mlp | text | probe_nonlinear | 67.73 | 67.36-68.06 | 67.71 | -0.34 | 0.0068 |
| modernbert-base (plain-first-natural) bilstm | text | probe_nonlinear | 67.72 | 67.34-68.05 | 67.70 | -0.35 | 0.0027 |
| bert-base-uncased (plain-first-natural) bilstm | text | probe_nonlinear | 67.70 | 67.33-68.02 | 67.69 | -0.37 | 0.0028 |
| qwen3.5-4b-base (echo-last-natural) bilstm | text | probe_nonlinear | 67.65 | 67.28-67.97 | 67.64 | -0.42 | 0.00061 |
| qwen3.5-2b-base (echo-last-natural) mlp | text | probe_nonlinear | 67.61 | 67.26-67.94 | 67.60 | -0.46 | 0.0002 |
| qwen3.5-0.8b-base (echo-last-natural) mlp | text | probe_nonlinear | 67.57 | 67.19-67.90 | 67.55 | -0.50 | 5.5e-05 |
| muse-glimmer-30b (echo-last-natural-train_100) bilstm | text | probe_nonlinear | 67.55 | 67.20-67.88 | 67.53 | -0.52 | 3.7e-05 |
| qwen3.5-4b-base (echo-last-natural) mlp | text | probe_nonlinear | 67.53 | 67.18-67.84 | 67.52 | -0.54 | 1.6e-05 |
| olmo-3-7b (echo-last-natural) mlp | text | probe_nonlinear | 67.49 | 67.14-67.81 | 67.46 | -0.58 | 3e-06 |
| bert-large-uncased (plain-first-natural) mlp | text | probe_nonlinear | 67.42 | 67.05-67.75 | 67.39 | -0.65 | 1.6e-07 |
| muse-glimmer-30b (echo-last-natural-train_100) mlp | text | probe_nonlinear | 67.42 | 67.04-67.74 | 67.41 | -0.65 | 5.1e-07 |
| modernbert-large (plain-first-natural) mlp | text | probe_nonlinear | 67.26 | 66.89-67.56 | 67.24 | -0.81 | 1.6e-12 |
| modernbert-base (plain-first-natural) mlp | text | probe_nonlinear | 67.23 | 66.86-67.56 | 67.21 | -0.84 | 6.6e-12 |
| bert-base-uncased (plain-first-natural-train_100) bilstm | text | probe_nonlinear | 67.20 | 66.86-67.53 | 67.18 | -0.87 | 6e-12 |
| bert-base-cased (plain-first-natural) mlp | text | probe_nonlinear | 67.15 | 66.82-67.51 | 67.13 | -0.92 | 7.2e-13 |
| xlm-roberta-base (plain-first-natural) mlp | text | probe_nonlinear | 67.14 | 66.77-67.47 | 67.11 | -0.93 | 1.3e-13 |
| xlm-roberta-large (plain-first-natural) mlp | text | probe_nonlinear | 67.12 | 66.77-67.47 | 67.10 | -0.95 | 3.7e-14 |
| muse-glimmer-30b (echo-last-natural) LDA | text | probe_linear | 66.92 | 66.56-67.25 | 66.91 | -1.15 | 2.6e-19 |
| bert-base-uncased (plain-first-natural) mlp | text | probe_nonlinear | 66.91 | 66.52-67.24 | 66.90 | -1.16 | 6.8e-19 |
| muse-glimmer-30b (echo-last-natural-train_100) LDA | text | probe_linear | 66.75 | 66.40-67.11 | 66.73 | -1.32 | 5.5e-24 |
| qwen3.8-27b (echo-last-natural) LDA | text | probe_linear | 66.75 | 66.39-67.08 | 66.72 | -1.32 | 2.1e-24 |
| olmo-3-32b (echo-last-natural) LDA | text | probe_linear | 66.74 | 66.37-67.08 | 66.70 | -1.33 | 5.5e-25 |
| qwen3.5-9b-base (echo-last-natural) LDA | text | probe_linear | 66.69 | 66.31-67.04 | 66.66 | -1.38 | 1.6e-26 |
| olmo-3-7b (echo-last-natural) LDA | text | probe_linear | 66.40 | 66.03-66.74 | 66.37 | -1.67 | 1.6e-37 |
| qwen3.5-4b-base (echo-last-natural) LDA | text | probe_linear | 66.33 | 65.96-66.63 | 66.29 | -1.74 | 1.8e-40 |
| qwen3.5-2b-base (echo-last-natural) LDA | text | probe_linear | 66.24 | 65.86-66.56 | 66.21 | -1.83 | 2e-44 |
| muse-glimmer-30b (echo-last-natural) logistic | text | probe_nonlinear | 66.22 | 65.86-66.53 | 66.20 | -1.85 | 1.6e-41 |
| muse-glimmer-30b (plain-last-natural) LDA | text | probe_linear | 66.16 | 65.82-66.48 | 66.13 | -1.91 | 9.6e-46 |
| muse-glimmer-30b (echo-last-natural-train_100) logistic | text | probe_nonlinear | 66.12 | 65.75-66.43 | 66.09 | -1.95 | 2.6e-45 |
| olmo-3-32b (echo-last-natural) logistic | text | probe_nonlinear | 66.12 | 65.78-66.44 | 66.08 | -1.95 | 8.9e-46 |
| bert-base-uncased (plain-first-natural-train_100) mlp | text | probe_nonlinear | 66.03 | 65.64-66.39 | 66.01 | -2.04 | 4.2e-52 |
| qwen3.5-9b-base (echo-last-natural) logistic | text | probe_nonlinear | 66.02 | 65.65-66.36 | 65.99 | -2.05 | 8.2e-51 |
| qwen3.5-2b-base (echo-last-natural) logistic | text | probe_nonlinear | 65.94 | 65.58-66.28 | 65.91 | -2.13 | 4.7e-56 |
| qwen3.5-4b-base (echo-last-natural) logistic | text | probe_nonlinear | 65.89 | 65.53-66.21 | 65.85 | -2.18 | 2.5e-58 |
| qwen3.8-27b (plain-last-natural) LDA | text | probe_linear | 65.86 | 65.49-66.17 | 65.82 | -2.21 | 2.8e-59 |
| qwen3.5-0.8b-base (echo-last-natural) logistic | text | probe_nonlinear | 65.81 | 65.44-66.16 | 65.79 | -2.26 | 4.4e-62 |
| qwen3.8-27b (echo-last-natural) logistic | text | probe_nonlinear | 65.78 | 65.42-66.11 | 65.75 | -2.29 | 1.2e-61 |
| qwen3.5-0.8b-base (echo-last-natural) LDA | text | probe_linear | 65.76 | 65.40-66.10 | 65.73 | -2.31 | 2.2e-66 |
| olmo-3-7b (echo-last-natural) logistic | text | probe_nonlinear | 65.76 | 65.42-66.08 | 65.73 | -2.31 | 1.1e-62 |
| qwen3.5-9b-base (plain-last-natural) LDA | text | probe_linear | 65.75 | 65.39-66.08 | 65.72 | -2.32 | 3.1e-65 |
| olmo-3-7b (plain-last-natural) LDA | text | probe_linear | 65.57 | 65.20-65.90 | 65.54 | -2.50 | 1.2e-73 |
| olmo-3-32b (plain-last-natural) LDA | text | probe_linear | 65.54 | 65.19-65.86 | 65.51 | -2.53 | 9.2e-75 |
| qwen3.5-4b-base (plain-last-natural) LDA | text | probe_linear | 65.54 | 65.16-65.85 | 65.50 | -2.53 | 1.6e-75 |
| xlm-roberta-large (plain-first-natural) logistic | text | probe_nonlinear | 65.37 | 65.02-65.70 | 65.34 | -2.70 | 3.3e-88 |
| qwen3.5-2b-base (plain-last-natural) LDA | text | probe_linear | 65.30 | 64.95-65.62 | 65.27 | -2.77 | 8.6e-90 |
| modernbert-large (plain-first-natural) LDA | text | probe_linear | 65.29 | 64.93-65.62 | 65.26 | -2.78 | 6e-99 |
| modernbert-base (plain-first-natural) logistic | text | probe_nonlinear | 65.27 | 64.91-65.61 | 65.24 | -2.80 | 6e-95 |
| xlm-roberta-large (plain-first-natural) LDA | text | probe_linear | 65.20 | 64.83-65.54 | 65.17 | -2.87 | 1.1e-98 |
| modernbert-large (plain-first-natural) logistic | text | probe_nonlinear | 65.18 | 64.81-65.50 | 65.15 | -2.89 | 6.5e-108 |
| xlm-roberta-base (plain-first-natural) logistic | text | probe_nonlinear | 65.17 | 64.83-65.51 | 65.14 | -2.90 | 5.6e-99 |
| bert-large-uncased (plain-first-natural) LDA | text | probe_linear | 65.08 | 64.70-65.40 | 65.04 | -2.99 | 6.6e-103 |
| bert-large-uncased (plain-first-natural) logistic | text | probe_nonlinear | 65.05 | 64.68-65.38 | 65.01 | -3.02 | 3.1e-105 |
| duration only | audio | baseline | 65.05 | 64.72-65.39 | 65.02 | -3.02 | 9.7e-70 |
| xlm-roberta-base (plain-first-natural) LDA | text | probe_linear | 64.98 | 64.62-65.32 | 64.95 | -3.09 | 4.6e-109 |
| modernbert-base (plain-first-natural) LDA | text | probe_linear | 64.95 | 64.59-65.29 | 64.92 | -3.12 | 3.7e-115 |
| muse-glimmer-30b (plain-last-natural-n5000) LDA | text | probe_linear | 64.94 | 64.58-65.27 | 64.91 | -3.13 | 3.2e-108 |
| bert-base-cased (plain-first-natural) logistic | text | probe_nonlinear | 64.85 | 64.50-65.16 | 64.82 | -3.22 | 3.6e-115 |
| bert-base-uncased (plain-first-natural) logistic | text | probe_nonlinear | 64.78 | 64.42-65.14 | 64.75 | -3.29 | 8.1e-120 |
| qwen3.5-0.8b-base (plain-last-natural) LDA | text | probe_linear | 64.74 | 64.39-65.09 | 64.71 | -3.33 | 8.7e-122 |
| bert-base-uncased (plain-first-natural-train_100) logistic | text | probe_nonlinear | 64.70 | 64.32-65.04 | 64.67 | -3.37 | 1.9e-125 |
| bert-base-uncased (plain-first-natural) LDA | text | probe_linear | 64.64 | 64.26-64.95 | 64.60 | -3.43 | 2.6e-128 |
| qwen3.8-27b (plain-last-natural-n5000) LDA | text | probe_linear | 64.61 | 64.26-64.95 | 64.58 | -3.46 | 1.5e-130 |
| bert-base-cased (plain-first-natural) LDA | text | probe_linear | 64.59 | 64.23-64.92 | 64.56 | -3.48 | 5.9e-133 |
| bert-base-uncased (plain-first-natural-train_100) LDA | text | probe_linear | 64.45 | 64.06-64.76 | 64.42 | -3.62 | 3.4e-141 |
| bert-base-uncased (plain-first-natural-n500) LDA | text | probe_linear | 62.76 | 62.40-63.08 | 62.73 | -5.31 | 1.4e-258 |
| qwen3.8-27b linguistic-k500 | text | icl | 62.03 | 61.66-62.35 | 62.01 | -6.04 | 8.3e-321 |
| deepseek-v4-pro linguistic-lines-3way-k5 | text | prompting | 60.54 | 60.14-60.92 | 60.52 | -7.53 | 0 |
| mistral-large-3 linguistic-lines-3way-k0 | text | prompting | 59.76 | 59.42-60.10 | 59.73 | -8.31 | 0 |
| gpt-4o linguistic-lines-3way-k5 | text | prompting | 59.47 | 59.08-59.83 | 59.44 | -8.60 | 0 |
| qwen2-audio-7b-instruct linguistic-no_audio | text | audio_prompting | 43.10 | 42.67-43.52 | 43.08 | -24.97 | 0 |
| qwen2-audio-7b-instruct linguistic-audio | audio | audio_prompting | 39.24 | 38.78-39.71 | 39.23 | -28.83 | 0 |

## 2-way

| system | modality | family | common words | 95% CI | all words | vs best text | p |
|---|---|---|---|---|---|---|---|
| wavelet toolkit (labelling procedure) | audio | reference | 95.00 | 94.84-95.15 | 94.93 | +11.92 | 0 |
| whisper-large-v3 (audio-mean) mlp | audio | probe_nonlinear | 91.06 | 90.84-91.27 | 90.98 | +7.98 | 0 |
| qwen2-audio-7b (audio-mean-encoder) bilstm | audio | probe_nonlinear | 91.01 | 90.80-91.22 | 90.93 | +7.93 | 0 |
| hubert-large (audio-mean) bilstm | audio | probe_nonlinear | 90.96 | 90.75-91.17 | 90.90 | +7.88 | 0 |
| whisper-large-v3 (audio-mean) bilstm | audio | probe_nonlinear | 90.87 | 90.64-91.08 | 90.80 | +7.79 | 0 |
| hubert-large (audio-mean) mlp | audio | probe_nonlinear | 90.69 | 90.47-90.91 | 90.63 | +7.61 | 0 |
| wavlm-large (audio-mean) bilstm | audio | probe_nonlinear | 90.66 | 90.44-90.87 | 90.60 | +7.58 | 0 |
| wavlm-large (audio-mean) mlp | audio | probe_nonlinear | 90.42 | 90.22-90.63 | 90.36 | +7.34 | 0 |
| qwen2-audio-7b (audio-mean-encoder) mlp | audio | probe_nonlinear | 90.37 | 90.16-90.59 | 90.29 | +7.29 | 0 |
| hubert-large (audio-mean) logistic | audio | probe_nonlinear | 88.93 | 88.69-89.16 | 88.87 | +5.85 | 0 |
| wavlm-large (audio-mean) logistic | audio | probe_nonlinear | 88.89 | 88.67-89.10 | 88.83 | +5.81 | 0 |
| whisper-large-v3 (audio-mean) logistic | audio | probe_nonlinear | 88.68 | 88.46-88.90 | 88.61 | +5.60 | 0 |
| qwen2-audio-7b (audio-mean-encoder) logistic | audio | probe_nonlinear | 88.50 | 88.27-88.73 | 88.43 | +5.42 | 0 |
| whisper-large-v3 (audio-mean) LDA | audio | probe_linear | 88.30 | 88.05-88.53 | 88.23 | +5.22 | 0 |
| hubert-large (audio-mean) LDA | audio | probe_linear | 88.29 | 88.06-88.52 | 88.24 | +5.21 | 0 |
| qwen2-audio-7b (audio-mean-encoder) LDA | audio | probe_linear | 88.16 | 87.92-88.40 | 88.09 | +5.08 | 0 |
| wavlm-large (audio-mean) LDA | audio | probe_linear | 87.89 | 87.64-88.15 | 87.83 | +4.81 | 0 |
| qwen2-audio-7b (audio-mean-llm) bilstm | audio | probe_nonlinear | 87.39 | 87.13-87.65 | 87.32 | +4.31 | 3.6e-314 |
| qwen2-audio-7b (audio-mean-llm) mlp | audio | probe_nonlinear | 87.27 | 87.00-87.53 | 87.20 | +4.19 | 6.8e-305 |
| qwen2-audio-7b (audio-mean-llm) LDA | audio | probe_linear | 86.52 | 86.26-86.80 | 86.45 | +3.44 | 8.2e-246 |
| qwen2-audio-7b (audio-mean-llm) logistic | audio | probe_nonlinear | 86.36 | 86.10-86.64 | 86.29 | +3.28 | 7.3e-188 |
| duration only | audio | baseline | 83.25 | 82.96-83.53 | 83.19 | +0.17 | 0.16 |
| modernbert-large (plain-first-natural) bilstm | text | probe_nonlinear | 83.08 | 82.75-83.39 | 83.06 |  |  |
| bert-base-cased (plain-first-natural) bilstm | text | probe_nonlinear | 83.05 | 82.73-83.35 | 83.02 | -0.03 | 0.65 |
| muse-glimmer-30b (echo-last-natural) mlp | text | probe_nonlinear | 83.02 | 82.70-83.33 | 83.00 | -0.06 | 0.44 |
| modernbert-base (plain-first-natural) bilstm | text | probe_nonlinear | 82.97 | 82.66-83.27 | 82.95 | -0.11 | 0.12 |
| bert-large-uncased (plain-first-natural) bilstm | text | probe_nonlinear | 82.96 | 82.64-83.26 | 82.94 | -0.12 | 0.11 |
| bert-base-uncased (plain-first-natural) bilstm | text | probe_nonlinear | 82.94 | 82.62-83.25 | 82.92 | -0.14 | 0.062 |
| qwen3.5-0.8b-base (echo-last-natural) bilstm | text | probe_nonlinear | 82.94 | 82.62-83.24 | 82.92 | -0.14 | 0.061 |
| qwen3.5-4b-base (echo-last-natural) bilstm | text | probe_nonlinear | 82.91 | 82.57-83.20 | 82.89 | -0.17 | 0.026 |
| qwen3.5-2b-base (echo-last-natural) bilstm | text | probe_nonlinear | 82.89 | 82.59-83.19 | 82.88 | -0.19 | 0.012 |
| olmo-3-7b (echo-last-natural) bilstm | text | probe_nonlinear | 82.88 | 82.55-83.19 | 82.86 | -0.20 | 0.014 |
| xlm-roberta-large (plain-first-natural) mlp | text | probe_nonlinear | 82.87 | 82.56-83.18 | 82.86 | -0.21 | 0.0075 |
| xlm-roberta-base (plain-first-natural) bilstm | text | probe_nonlinear | 82.86 | 82.54-83.17 | 82.85 | -0.22 | 0.0035 |
| muse-glimmer-30b (echo-last-natural) bilstm | text | probe_nonlinear | 82.85 | 82.54-83.15 | 82.83 | -0.23 | 0.0036 |
| bert-base-uncased (plain-first-natural) mlp | text | probe_nonlinear | 82.83 | 82.50-83.15 | 82.81 | -0.25 | 0.0018 |
| bert-large-uncased (plain-first-natural) mlp | text | probe_nonlinear | 82.82 | 82.50-83.12 | 82.80 | -0.26 | 0.00065 |
| qwen3.8-27b (echo-last-natural) bilstm | text | probe_nonlinear | 82.82 | 82.51-83.13 | 82.80 | -0.26 | 0.00085 |
| xlm-roberta-large (plain-first-natural) bilstm | text | probe_nonlinear | 82.82 | 82.50-83.14 | 82.81 | -0.26 | 0.00061 |
| bert-base-cased (plain-first-natural) mlp | text | probe_nonlinear | 82.81 | 82.49-83.12 | 82.79 | -0.27 | 0.00055 |
| qwen3.8-27b (echo-last-natural) mlp | text | probe_nonlinear | 82.80 | 82.49-83.10 | 82.77 | -0.28 | 0.00057 |
| qwen3.5-9b-base (echo-last-natural) bilstm | text | probe_nonlinear | 82.79 | 82.49-83.08 | 82.78 | -0.29 | 0.00024 |
| muse-glimmer-30b (echo-last-natural-train_100) mlp | text | probe_nonlinear | 82.78 | 82.46-83.09 | 82.76 | -0.30 | 0.00016 |
| olmo-3-32b (echo-last-natural) bilstm | text | probe_nonlinear | 82.77 | 82.47-83.06 | 82.74 | -0.31 | 7.1e-05 |
| qwen3.5-9b-base (echo-last-natural) mlp | text | probe_nonlinear | 82.77 | 82.45-83.07 | 82.75 | -0.31 | 0.00011 |
| muse-glimmer-30b (echo-last-natural-train_100) bilstm | text | probe_nonlinear | 82.75 | 82.43-83.08 | 82.73 | -0.33 | 4.4e-05 |
| modernbert-base (plain-first-natural) mlp | text | probe_nonlinear | 82.74 | 82.42-83.05 | 82.73 | -0.34 | 8.2e-06 |
| qwen3.5-2b-base (echo-last-natural) mlp | text | probe_nonlinear | 82.74 | 82.43-83.04 | 82.72 | -0.34 | 1.8e-05 |
| bert-base-uncased (plain-first-natural-train_100) bilstm | text | probe_nonlinear | 82.73 | 82.39-83.04 | 82.71 | -0.35 | 5.8e-06 |
| modernbert-large (plain-first-natural) mlp | text | probe_nonlinear | 82.63 | 82.31-82.94 | 82.60 | -0.45 | 3.1e-10 |
| olmo-3-32b (echo-last-natural) mlp | text | probe_nonlinear | 82.63 | 82.32-82.93 | 82.60 | -0.45 | 7.1e-08 |
| qwen3.5-0.8b-base (echo-last-natural) mlp | text | probe_nonlinear | 82.63 | 82.30-82.95 | 82.61 | -0.45 | 1.7e-08 |
| qwen3.5-4b-base (echo-last-natural) mlp | text | probe_nonlinear | 82.63 | 82.32-82.96 | 82.61 | -0.45 | 1.9e-08 |
| xlm-roberta-base (plain-first-natural) mlp | text | probe_nonlinear | 82.62 | 82.29-82.94 | 82.60 | -0.46 | 3.3e-09 |
| muse-glimmer-30b (echo-last-natural) LDA | text | probe_linear | 82.61 | 82.29-82.90 | 82.59 | -0.47 | 1.7e-09 |
| olmo-3-7b (echo-last-natural) mlp | text | probe_nonlinear | 82.57 | 82.25-82.87 | 82.54 | -0.51 | 5.5e-10 |
| muse-glimmer-30b (echo-last-natural-train_100) LDA | text | probe_linear | 82.52 | 82.20-82.83 | 82.50 | -0.56 | 2.6e-12 |
| bert-base-uncased (plain-first-natural-train_100) mlp | text | probe_nonlinear | 82.52 | 82.19-82.83 | 82.49 | -0.56 | 1.4e-12 |
| qwen3.8-27b (echo-last-natural) LDA | text | probe_linear | 82.37 | 82.06-82.68 | 82.35 | -0.71 | 2.5e-18 |
| muse-glimmer-30b (plain-last-natural) LDA | text | probe_linear | 82.22 | 81.90-82.53 | 82.19 | -0.86 | 6e-25 |
| olmo-3-32b (echo-last-natural) LDA | text | probe_linear | 82.20 | 81.88-82.51 | 82.17 | -0.88 | 6.6e-26 |
| muse-glimmer-30b (echo-last-natural) logistic | text | probe_nonlinear | 82.16 | 81.86-82.48 | 82.15 | -0.92 | 2.7e-25 |
| qwen3.5-9b-base (echo-last-natural) LDA | text | probe_linear | 82.15 | 81.81-82.47 | 82.12 | -0.93 | 7.1e-29 |
| olmo-3-7b (echo-last-natural) LDA | text | probe_linear | 82.14 | 81.82-82.45 | 82.11 | -0.94 | 2e-29 |
| muse-glimmer-30b (echo-last-natural-train_100) logistic | text | probe_nonlinear | 82.13 | 81.82-82.47 | 82.10 | -0.95 | 6.8e-27 |
| qwen3.5-4b-base (echo-last-natural) LDA | text | probe_linear | 82.12 | 81.78-82.45 | 82.09 | -0.96 | 3.5e-31 |
| xlm-roberta-large (plain-first-natural) LDA | text | probe_linear | 82.06 | 81.75-82.38 | 82.04 | -1.02 | 2.6e-34 |
| qwen3.5-2b-base (echo-last-natural) LDA | text | probe_linear | 82.05 | 81.73-82.37 | 82.03 | -1.03 | 2e-34 |
| qwen3.8-27b (echo-last-natural) logistic | text | probe_nonlinear | 82.04 | 81.71-82.36 | 82.01 | -1.04 | 6.6e-31 |
| xlm-roberta-large (plain-first-natural) logistic | text | probe_nonlinear | 82.03 | 81.72-82.34 | 82.01 | -1.05 | 1.2e-34 |
| bert-large-uncased (plain-first-natural) LDA | text | probe_linear | 82.00 | 81.67-82.33 | 81.96 | -1.08 | 3.6e-37 |
| qwen3.5-2b-base (echo-last-natural) logistic | text | probe_nonlinear | 82.00 | 81.66-82.32 | 81.97 | -1.08 | 5.4e-35 |
| olmo-3-32b (plain-last-natural) LDA | text | probe_linear | 81.97 | 81.62-82.28 | 81.92 | -1.11 | 2.6e-38 |
| qwen3.5-4b-base (echo-last-natural) logistic | text | probe_nonlinear | 81.97 | 81.65-82.30 | 81.95 | -1.11 | 7.6e-37 |
| muse-glimmer-30b (plain-last-natural-n5000) LDA | text | probe_linear | 81.96 | 81.61-82.29 | 81.91 | -1.12 | 3.5e-38 |
| bert-large-uncased (plain-first-natural) logistic | text | probe_nonlinear | 81.96 | 81.63-82.30 | 81.92 | -1.12 | 3.5e-38 |
| qwen3.8-27b (plain-last-natural) LDA | text | probe_linear | 81.92 | 81.60-82.24 | 81.90 | -1.16 | 4.6e-41 |
| bert-base-uncased (plain-first-natural) logistic | text | probe_nonlinear | 81.92 | 81.59-82.25 | 81.88 | -1.16 | 1.4e-41 |
| xlm-roberta-base (plain-first-natural) logistic | text | probe_nonlinear | 81.92 | 81.60-82.24 | 81.90 | -1.16 | 8.5e-42 |
| modernbert-large (plain-first-natural) LDA | text | probe_linear | 81.91 | 81.60-82.23 | 81.86 | -1.17 | 6.1e-46 |
| bert-base-uncased (plain-first-natural) LDA | text | probe_linear | 81.89 | 81.56-82.21 | 81.84 | -1.19 | 2.7e-44 |
| bert-base-cased (plain-first-natural) logistic | text | probe_nonlinear | 81.89 | 81.56-82.21 | 81.85 | -1.19 | 9.1e-43 |
| bert-base-cased (plain-first-natural) LDA | text | probe_linear | 81.87 | 81.56-82.20 | 81.83 | -1.21 | 2.2e-45 |
| olmo-3-7b (plain-last-natural) LDA | text | probe_linear | 81.87 | 81.52-82.19 | 81.82 | -1.21 | 1e-44 |
| qwen3.5-0.8b-base (echo-last-natural) logistic | text | probe_nonlinear | 81.87 | 81.55-82.18 | 81.84 | -1.21 | 1.1e-42 |
| bert-base-uncased (plain-first-natural-train_100) LDA | text | probe_linear | 81.86 | 81.53-82.19 | 81.81 | -1.22 | 3.2e-46 |
| olmo-3-32b (echo-last-natural) logistic | text | probe_nonlinear | 81.86 | 81.56-82.19 | 81.83 | -1.22 | 1.8e-40 |
| qwen3.5-9b-base (echo-last-natural) logistic | text | probe_nonlinear | 81.86 | 81.54-82.17 | 81.84 | -1.22 | 4.2e-42 |
| xlm-roberta-base (plain-first-natural) LDA | text | probe_linear | 81.85 | 81.53-82.16 | 81.82 | -1.23 | 7.6e-47 |
| qwen3.5-9b-base (plain-last-natural) LDA | text | probe_linear | 81.84 | 81.52-82.15 | 81.82 | -1.24 | 7.4e-47 |
| olmo-3-7b (echo-last-natural) logistic | text | probe_nonlinear | 81.83 | 81.51-82.14 | 81.81 | -1.25 | 1.7e-43 |
| bert-base-uncased (plain-first-natural-train_100) logistic | text | probe_nonlinear | 81.80 | 81.48-82.12 | 81.76 | -1.28 | 1.9e-49 |
| modernbert-large (plain-first-natural) logistic | text | probe_nonlinear | 81.80 | 81.48-82.11 | 81.77 | -1.28 | 1.4e-51 |
| qwen3.5-0.8b-base (echo-last-natural) LDA | text | probe_linear | 81.77 | 81.45-82.10 | 81.74 | -1.31 | 8.1e-51 |
| modernbert-base (plain-first-natural) logistic | text | probe_nonlinear | 81.77 | 81.45-82.09 | 81.75 | -1.31 | 8.5e-49 |
| qwen3.5-4b-base (plain-last-natural) LDA | text | probe_linear | 81.74 | 81.41-82.06 | 81.71 | -1.34 | 2.8e-54 |
| qwen3.5-2b-base (plain-last-natural) LDA | text | probe_linear | 81.72 | 81.39-82.05 | 81.70 | -1.36 | 7.9e-55 |
| modernbert-base (plain-first-natural) LDA | text | probe_linear | 81.68 | 81.35-82.00 | 81.66 | -1.40 | 7.9e-58 |
| qwen3.5-0.8b-base (plain-last-natural) LDA | text | probe_linear | 81.55 | 81.23-81.86 | 81.54 | -1.53 | 1.3e-65 |
| qwen3.8-27b (plain-last-natural-n5000) LDA | text | probe_linear | 81.53 | 81.22-81.85 | 81.50 | -1.55 | 6.9e-66 |
| bert-base-uncased (plain-first-natural-n500) LDA | text | probe_linear | 81.45 | 81.14-81.78 | 81.41 | -1.63 | 1.1e-72 |
| gpt-4o linguistic-lines-3way-k5 (derived) | text | prompting | 79.71 | 79.34-80.05 | 79.67 | -3.37 | 1.4e-218 |
| deepseek-v4-pro linguistic-lines-3way-k5 (derived) | text | prompting | 79.25 | 78.89-79.59 | 79.21 | -3.83 | 3.8e-261 |
| mistral-large-3 linguistic-lines-3way-k0 (derived) | text | prompting | 78.98 | 78.61-79.29 | 78.93 | -4.10 | 4.2e-291 |
| qwen3.8-27b linguistic-k500 (derived) | text | icl | 78.34 | 77.98-78.66 | 78.30 | -4.74 | 0 |
| qwen2-audio-7b-instruct linguistic-no_audio (derived) | text | audio_prompting | 51.61 | 51.29-51.93 | 51.60 | -31.47 | 0 |
| qwen2-audio-7b-instruct linguistic-audio (derived) | audio | audio_prompting | 50.98 | 50.68-51.33 | 50.97 | -32.10 | 0 |
