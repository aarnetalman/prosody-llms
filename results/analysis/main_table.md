# Main results (test, 2019 subset)

Accuracy on scored words; mean ± sd over seeds where there are several. Best layers and best models chosen on dev.

| group | system | modality | 2-way | 3-way | note |
|---|---|---|---|---|---|
| Baselines | Majority class | text | 51.98 | 48.02 |  |
| Baselines | Word majority (2019) | text | 80.69 | 61.96 |  |
| Fine-tuned | BERT-base (2019 recipe) | text | 83.02 ± 0.13 | 68.49 ± 0.10 |  |
| Fine-tuned | BERT-large | text | 83.15 ± 0.09 | 68.54 ± 0.13 |  |
| Fine-tuned | ModernBERT-large | text | 83.22 ± 0.05 | 68.80 ± 0.08 |  |
| Fine-tuned | XLM-R large | text | 83.28 ± 0.11 | 68.80 ± 0.17 | 1 of 3 seeds did not train (dev at the majority-class level), left out |
| Frozen, linear probe | BERT-base | text | 81.84 | 64.61 | layer 4/12 |
| Frozen, linear probe | best text: Muse Glimmer 30B, echo | text | 82.59 | 66.91 | layer 10/52 |
| Frozen, linear probe | best audio: Whisper large-v3 (encoder) | audio | 88.22 | 78.74 | layer 23/32 |
| Frozen, BiLSTM classifier | best text (3-way): Qwen3.5 0.8B, echo | text | 82.92 ± 0.03 | 67.98 ± 0.10 | layer 15 |
| Frozen, BiLSTM classifier | best text (2-way): ModernBERT-large | text | 83.03 ± 0.03 | 68.00 ± 0.06 | layer 25; layer 19 |
| Frozen, BiLSTM classifier | best audio: Whisper large-v3 (encoder) | audio | 90.99 ± 0.19 | 84.66 ± 0.16 | layer 23 |
| Prompting | GPT-4o | text | 79.67 | 59.44 |  |
| Prompting | DeepSeek-V4-Pro | text | 79.21 | 60.52 |  |
| Prompting | Mistral Large 3 | text | 78.93 | 59.74 |  |
| Prompting | GPT-5.5$^{\dagger}$ | text | 78.90 | 58.75 | test200 |
| Prompting | Qwen3.8 27B, 500 examples | text | 78.30 | 62.01 |  |
| Prompting | Qwen2-Audio, with recording | audio | 50.98 | 39.23 |  |
| Prompting | Qwen2-Audio, text only | text | 51.61 | 43.09 |  |
| Audio references | Duration only | audio | 83.20 | 65.02 |  |
| Audio references | Labelling procedure | audio | 94.93 | 91.44 |  |
