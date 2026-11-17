# Error analysis

**Agreement between readers of the same text** (390 texts, 6,277 word pairs): 77.0 (2-way), 63.8 (3-way).

## Do the text models agree with each other?

42 text probes on the whole test set. Mean accuracy 66.5 (3-way) / 82.0 (2-way); mean agreement between two models 82.6 / 94.0. All models give the same label to 45.8% / 74.4% of the words, and the same wrong label to 6.3% / 8.1%.

## Text-only ceiling from other readers (exploratory: too few texts)

Texts read by at least three speakers: 5 texts, 20 readings, 124 words. Each reading predicted from the other readers' mean prominence score: 82.3 (2-way), 75.0 (3-way).


## Test utterances whose text also occurs in train_360 or dev 

189 utterances, 2,733 scored words.

| system | task | overlap | rest | difference |
|---|---|---|---|---|
| bert-base-cased (plain-first-natural) BiLSTM | 2 | 77.97 | 82.84 | -4.87 |
| muse-glimmer-30b (echo-last-natural) BiLSTM | 2 | 79.62 | 82.83 | -3.21 |
| bert-large-uncased (plain-first-natural) BiLSTM | 2 | 79.03 | 82.76 | -3.72 |
| qwen3.5-0.8b-base (echo-last-natural) BiLSTM | 2 | 78.85 | 82.75 | -3.90 |
| modernbert-base (plain-first-natural) BiLSTM | 2 | 79.03 | 82.75 | -3.71 |
| modernbert-large (plain-first-natural) BiLSTM | 2 | 79.14 | 82.71 | -3.57 |
| muse-glimmer-30b (echo-last-natural-train_100) BiLSTM | 2 | 78.78 | 82.70 | -3.92 |
| xlm-roberta-base (plain-first-natural) BiLSTM | 2 | 78.92 | 82.62 | -3.69 |
| olmo-3-7b (echo-last-natural) BiLSTM | 2 | 78.92 | 82.60 | -3.68 |
| qwen3.5-4b-base (echo-last-natural) BiLSTM | 2 | 79.03 | 82.56 | -3.53 |
| olmo-3-32b (echo-last-natural) BiLSTM | 2 | 78.49 | 82.56 | -4.08 |
| qwen3.5-9b-base (echo-last-natural) BiLSTM | 2 | 78.56 | 82.56 | -4.00 |
| qwen3.8-27b (echo-last-natural) BiLSTM | 2 | 78.52 | 82.56 | -4.04 |
| xlm-roberta-large (plain-first-natural) BiLSTM | 2 | 79.25 | 82.49 | -3.24 |
| qwen3.5-2b-base (echo-last-natural) BiLSTM | 2 | 79.07 | 82.47 | -3.40 |
| bert-base-uncased (plain-first-natural-train_100) BiLSTM | 2 | 78.74 | 82.44 | -3.70 |
| muse-glimmer-30b (echo-last-natural) LDA | 2 | 78.63 | 82.43 | -3.80 |
| bert-base-uncased (plain-first-natural) BiLSTM | 2 | 78.78 | 82.43 | -3.65 |
| muse-glimmer-30b (echo-last-natural-train_100) LDA | 2 | 78.08 | 82.35 | -4.26 |
| qwen3.8-27b (echo-last-natural) LDA | 2 | 78.30 | 82.17 | -3.87 |
| muse-glimmer-30b (plain-last-natural) LDA | 2 | 77.61 | 82.13 | -4.53 |
| olmo-3-32b (echo-last-natural) LDA | 2 | 77.64 | 82.10 | -4.46 |
| qwen3.5-9b-base (echo-last-natural) LDA | 2 | 77.79 | 82.10 | -4.31 |
| qwen3.5-2b-base (echo-last-natural) LDA | 2 | 77.42 | 81.98 | -4.56 |
| olmo-3-7b (echo-last-natural) LDA | 2 | 77.64 | 81.94 | -4.29 |
| qwen3.5-4b-base (echo-last-natural) LDA | 2 | 77.46 | 81.94 | -4.47 |
| qwen3.8-27b (plain-last-natural) LDA | 2 | 77.61 | 81.88 | -4.27 |
| xlm-roberta-large (plain-first-natural) LDA | 2 | 77.02 | 81.85 | -4.83 |
| bert-large-uncased (plain-first-natural) LDA | 2 | 76.66 | 81.82 | -5.16 |
| modernbert-large (plain-first-natural) LDA | 2 | 76.58 | 81.77 | -5.19 |
| qwen3.5-9b-base (plain-last-natural) LDA | 2 | 77.64 | 81.77 | -4.13 |
| qwen3.5-2b-base (plain-last-natural) LDA | 2 | 77.13 | 81.69 | -4.56 |
| qwen3.5-0.8b-base (echo-last-natural) LDA | 2 | 76.66 | 81.67 | -5.02 |
| bert-base-cased (plain-first-natural) LDA | 2 | 76.69 | 81.65 | -4.96 |
| bert-base-uncased (plain-first-natural-train_100) LDA | 2 | 76.66 | 81.63 | -4.97 |
| qwen3.5-4b-base (plain-last-natural) LDA | 2 | 76.95 | 81.62 | -4.68 |
| bert-base-uncased (plain-first-natural) LDA | 2 | 76.84 | 81.62 | -4.78 |
| xlm-roberta-base (plain-first-natural) LDA | 2 | 76.95 | 81.59 | -4.64 |
| olmo-3-7b (plain-last-natural) LDA | 2 | 77.02 | 81.51 | -4.49 |
| modernbert-base (plain-first-natural) LDA | 2 | 76.51 | 81.51 | -5.00 |
| olmo-3-32b (plain-last-natural) LDA | 2 | 76.88 | 81.44 | -4.56 |
| qwen3.5-0.8b-base (plain-last-natural) LDA | 2 | 76.44 | 81.27 | -4.84 |
| gpt-4o linguistic-lines-3way-k5 | 2 | 74.02 | 79.85 | -5.82 |
| deepseek-v4-pro linguistic-lines-3way-k5 | 2 | 74.46 | 79.36 | -4.90 |
| mistral-large-3 linguistic-lines-3way-k0 | 2 | 74.35 | 79.07 | -4.72 |
| qwen3.8-27b linguistic-k0 | 2 | 74.53 | 78.93 | -4.40 |
| qwen3.8-27b linguistic-k500 | 2 | 74.46 | 78.42 | -3.96 |
| modernbert-large (plain-first-natural) BiLSTM | 3 | 65.72 | 68.13 | -2.41 |
| qwen3.5-0.8b-base (echo-last-natural) BiLSTM | 3 | 65.75 | 68.12 | -2.37 |
| xlm-roberta-large (plain-first-natural) BiLSTM | 3 | 65.90 | 68.00 | -2.11 |
| bert-large-uncased (plain-first-natural) BiLSTM | 3 | 65.72 | 67.98 | -2.26 |
| muse-glimmer-30b (echo-last-natural) BiLSTM | 3 | 66.08 | 67.95 | -1.87 |
| olmo-3-32b (echo-last-natural) BiLSTM | 3 | 64.98 | 67.90 | -2.92 |
| qwen3.5-9b-base (echo-last-natural) BiLSTM | 3 | 64.95 | 67.88 | -2.94 |
| qwen3.5-2b-base (echo-last-natural) BiLSTM | 3 | 66.26 | 67.88 | -1.62 |
| olmo-3-7b (echo-last-natural) BiLSTM | 3 | 65.13 | 67.87 | -2.74 |
| bert-base-cased (plain-first-natural) BiLSTM | 3 | 64.65 | 67.85 | -3.19 |
| qwen3.8-27b (echo-last-natural) BiLSTM | 3 | 65.28 | 67.84 | -2.57 |
| xlm-roberta-base (plain-first-natural) BiLSTM | 3 | 65.57 | 67.81 | -2.24 |
| modernbert-base (plain-first-natural) BiLSTM | 3 | 65.13 | 67.78 | -2.65 |
| bert-base-uncased (plain-first-natural) BiLSTM | 3 | 65.31 | 67.77 | -2.45 |
| qwen3.5-4b-base (echo-last-natural) BiLSTM | 3 | 64.73 | 67.73 | -3.01 |
| muse-glimmer-30b (echo-last-natural-train_100) BiLSTM | 3 | 64.73 | 67.62 | -2.89 |
| bert-base-uncased (plain-first-natural-train_100) BiLSTM | 3 | 64.91 | 67.25 | -2.34 |
| muse-glimmer-30b (echo-last-natural) LDA | 3 | 63.34 | 67.02 | -3.68 |
| muse-glimmer-30b (echo-last-natural-train_100) LDA | 3 | 63.19 | 66.84 | -3.65 |
| olmo-3-32b (echo-last-natural) LDA | 3 | 63.01 | 66.82 | -3.81 |
| qwen3.8-27b (echo-last-natural) LDA | 3 | 64.11 | 66.80 | -2.70 |
| qwen3.5-9b-base (echo-last-natural) LDA | 3 | 63.70 | 66.75 | -3.05 |
| olmo-3-7b (echo-last-natural) LDA | 3 | 62.79 | 66.48 | -3.69 |
| qwen3.5-4b-base (echo-last-natural) LDA | 3 | 62.39 | 66.41 | -4.02 |
| qwen3.5-2b-base (echo-last-natural) LDA | 3 | 63.01 | 66.31 | -3.30 |
| muse-glimmer-30b (plain-last-natural) LDA | 3 | 62.68 | 66.24 | -3.56 |
| qwen3.8-27b (plain-last-natural) LDA | 3 | 62.50 | 65.93 | -3.43 |
| qwen3.5-0.8b-base (echo-last-natural) LDA | 3 | 61.80 | 65.85 | -4.05 |
| qwen3.5-9b-base (plain-last-natural) LDA | 3 | 63.41 | 65.79 | -2.38 |
| olmo-3-7b (plain-last-natural) LDA | 3 | 61.76 | 65.66 | -3.90 |
| qwen3.5-4b-base (plain-last-natural) LDA | 3 | 61.54 | 65.63 | -4.08 |
| olmo-3-32b (plain-last-natural) LDA | 3 | 61.91 | 65.62 | -3.71 |
| qwen3.5-2b-base (plain-last-natural) LDA | 3 | 61.87 | 65.38 | -3.51 |
| modernbert-large (plain-first-natural) LDA | 3 | 61.95 | 65.36 | -3.41 |
| xlm-roberta-large (plain-first-natural) LDA | 3 | 61.73 | 65.28 | -3.55 |
| bert-large-uncased (plain-first-natural) LDA | 3 | 60.48 | 65.18 | -4.70 |
| xlm-roberta-base (plain-first-natural) LDA | 3 | 61.54 | 65.05 | -3.51 |
| modernbert-base (plain-first-natural) LDA | 3 | 61.54 | 65.02 | -3.48 |
| qwen3.5-0.8b-base (plain-last-natural) LDA | 3 | 61.00 | 64.83 | -3.84 |
| bert-base-uncased (plain-first-natural) LDA | 3 | 60.78 | 64.72 | -3.95 |
| bert-base-cased (plain-first-natural) LDA | 3 | 60.34 | 64.69 | -4.35 |
| bert-base-uncased (plain-first-natural-train_100) LDA | 3 | 60.30 | 64.55 | -4.25 |
| qwen3.8-27b linguistic-k500 | 3 | 59.64 | 62.08 | -2.44 |
| deepseek-v4-pro linguistic-lines-3way-k5 | 3 | 56.42 | 60.65 | -4.22 |
| mistral-large-3 linguistic-lines-3way-k0 | 3 | 56.46 | 59.84 | -3.38 |
| gpt-4o linguistic-lines-3way-k5 | 3 | 54.12 | 59.61 | -5.49 |
| qwen3.8-27b linguistic-k0 | 3 | 55.80 | 58.71 | -2.91 |

## Best text against best audio system on the common words

### 3-way: modernbert-large (plain-first-natural) bilstm against whisper-large-v3 (audio-mean) bilstm

| dimension | value | words | text | audio | audio - text | audio right, text wrong |
|---|---|---|---|---|---|---|
| all | all | 89,935 | 68.07 | 84.60 | +16.53 | 22.05 |
| class | content | 42,281 | 56.08 | 79.02 | +22.94 | 30.65 |
| class | function | 47,654 | 78.71 | 89.55 | +10.84 | 14.41 |
| frequency | 1-10 | 4,385 | 58.86 | 80.09 | +21.23 | 29.14 |
| frequency | 101-1k | 18,332 | 55.63 | 79.03 | +23.40 | 31.04 |
| frequency | 11-100 | 12,236 | 56.46 | 78.60 | +22.14 | 29.66 |
| frequency | >1k | 53,641 | 75.92 | 88.42 | +12.51 | 16.54 |
| frequency | unseen | 1,341 | 60.33 | 77.26 | +16.93 | 26.77 |
| givenness | - | 47,654 | 78.71 | 89.55 | +10.84 | 14.41 |
| givenness | given | 12,652 | 55.30 | 78.78 | +23.48 | 31.43 |
| givenness | new | 29,629 | 56.42 | 79.12 | +22.70 | 30.32 |
| gold | 0 | 43,186 | 83.56 | 90.37 | +6.81 | 10.35 |
| gold | 1 | 24,503 | 45.28 | 72.52 | +27.24 | 36.53 |
| gold | 2 | 22,246 | 63.10 | 86.71 | +23.60 | 28.80 |
| position | final | 4,801 | 51.70 | 75.13 | +23.43 | 32.33 |
| position | initial | 4,740 | 67.03 | 83.38 | +16.35 | 21.46 |
| position | medial | 80,394 | 69.11 | 85.24 | +16.13 | 21.47 |
| text consensus | all-right | 35,611 | 100.00 | 97.10 | -2.90 | 0.00 |
| text consensus | all-wrong | 9,699 | 0.00 | 58.93 | +58.93 | 58.93 |
| text consensus | split | 44,625 | 57.38 | 80.20 | +22.81 | 31.62 |

### 2-way: modernbert-large (plain-first-natural) bilstm against whisper-large-v3 (audio-mean) mlp

| dimension | value | words | text | audio | audio - text | audio right, text wrong |
|---|---|---|---|---|---|---|
| all | all | 89,935 | 83.08 | 91.06 | +7.98 | 11.47 |
| class | content | 42,281 | 83.01 | 89.96 | +6.95 | 10.61 |
| class | function | 47,654 | 83.14 | 92.03 | +8.89 | 12.23 |
| frequency | 1-10 | 4,385 | 89.05 | 92.61 | +3.56 | 6.02 |
| frequency | 101-1k | 18,332 | 81.51 | 89.56 | +8.05 | 11.90 |
| frequency | 11-100 | 12,236 | 85.73 | 90.63 | +4.90 | 7.71 |
| frequency | >1k | 53,641 | 82.35 | 91.54 | +9.19 | 12.78 |
| frequency | unseen | 1,341 | 90.08 | 90.98 | +0.89 | 5.22 |
| givenness | - | 47,654 | 83.14 | 92.03 | +8.89 | 12.23 |
| givenness | given | 12,652 | 79.26 | 88.54 | +9.28 | 13.63 |
| givenness | new | 29,629 | 84.61 | 90.56 | +5.95 | 9.32 |
| gold | 0 | 43,186 | 79.31 | 89.76 | +10.45 | 13.45 |
| gold | 1 | 46,749 | 86.56 | 92.25 | +5.69 | 9.64 |
| position | final | 4,801 | 78.32 | 86.34 | +8.02 | 13.33 |
| position | initial | 4,740 | 78.23 | 89.24 | +11.01 | 16.62 |
| position | medial | 80,394 | 83.65 | 91.45 | +7.80 | 11.05 |

## Examples (3-way; word[gold|text|audio]; first five by a fixed hash)

- `7176_92135_000035_000001`: You[1|0|0] will[2|0|2] see[0|2|0] this[0|0|0] for[1|0|0] yourself[0|2|2] if[1|0|1] you[0|0|0] consider[2|2|2] the[0|0|0] passage[0|2|1] as[2|1|2] it[0|0|0] should[2|0|2] properly[1|2|1] have[0|0|0] been[0|0|0] written[0|1|1]
- `121_127105_000027_000000`: Well[2|2|2] then[0|2|0] ; after[0|1|0] dinner[2|1|2]
- `4970_29095_000032_000001`: If[1|0|1] thee[2|2|2] wants[1|1|2] to[0|0|0] see[1|0|1] new[1|1|0] bonnets[1|1|1] , orthodox[2|2|2] to[0|0|0] a[0|0|0] shade[1|2|1] and[0|0|0] conformed[2|2|2] to[0|0|0] the[0|0|0] letter[1|1|1] of[0|0|0] the[0|0|0] true[1|1|1] form[2|2|2] , thee[1|1|1] must[1|2|2] go[1|0|1] to[0|0|0] the[0|0|0] Arch[0|1|1] Street[0|0|0] Meeting[1|0|1] .
- `1089_134686_000023_000000`: It[1|0|1] was[0|0|0] strange[2|2|2] too[1|2|1] that[0|0|0] he[0|0|0] found[2|2|2] an[0|0|0] arid[2|2|2] pleasure[2|1|2] in[0|0|0] following[2|2|2] up[1|0|1] to[0|0|0] the[0|0|0] end[2|1|1] the[0|0|0] rigid[1|2|1] lines[1|1|1] of[0|0|0] the[0|0|0] doctrines[1|2|1] of[0|0|0] the[0|0|0] church[2|1|2] and[0|0|0] penetrating[2|2|2] into[0|0|0] obscure[2|1|1] silences[2|1|2] only[2|1|2] to[0|0|0] hear[2|2|2] and[0|0|0] feel[2|2|2] the[0|0|0] more[0|0|0] deeply[2|1|2] his[0|0|0] own[2|1|1] condemnation[1|1|1] .
- `7127_75946_000002_000000`: At[1|1|1] the[0|0|0] conclusion[2|2|2] of[0|0|0] the[0|0|0] banquet[1|1|1] , which[1|1|1] was[0|0|0] served[2|2|2] at[0|0|0] five[1|1|1] o'clock[1|2|2] , the[0|0|0] king[2|2|2] entered[0|1|0] his[0|0|0] cabinet[2|1|2] , where[0|0|0] his[0|0|0] tailors[2|2|2] were[0|0|0] awaiting[2|2|2] him[0|0|0] for[0|0|0] the[0|0|0] purpose[2|2|2] of[0|0|0] trying[2|2|2] on[1|0|1] the[0|0|0] celebrated[2|1|1] costume[2|1|2] representing[0|1|1] Spring[2|2|2] , which[0|1|2] was[2|0|0] the[0|0|0] result[1|2|2] of[0|0|0] so[0|2|1] much[1|0|1] imagination[1|2|1] , and[2|0|2] had[0|0|0] cost[2|1|1] so[1|2|1] many[1|0|2] efforts[0|1|0] of[0|0|0] thought[2|2|2] to[0|0|0] the[0|0|0] designers[2|2|2] and[0|0|0] ornament[1|1|1] workers[1|0|0] of[0|0|0] the[0|0|0] court[1|1|2] .
