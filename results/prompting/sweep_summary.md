# Prompting sweep

Accuracy on scored words (punctuation excluded). 2-way from a 3-way prompt is derived; `2way` prompts are native. Selection: best 3-way accuracy on the dev subset.

## Selected settings

- gpt-4o: linguistic / lines / 5 shots
- deepseek-v4-pro: linguistic / lines / 5 shots
- mistral-large-3: linguistic / lines / 0 shots

## Results

| model | split | prompt | 3-way | 2-way | AUC | coverage | missing |
|---|---|---|---|---|---|---|---|
| gpt-4o | dev500 | linguistic-json-3way-k0 | 57.82 | 74.10 |  | 1.0000 | 0 |
| gpt-4o | dev500 | linguistic-lines-2way-k5 |  | 80.29 | 0.853 | 1.0000 | 0 |
| gpt-4o | dev500 | linguistic-lines-3way-k0 | 59.33 | 79.18 | 0.846 | 0.9967 | 0 |
| gpt-4o | dev500 | linguistic-lines-3way-k1 | 59.77 | 79.40 | 0.847 | 0.9967 | 0 |
| gpt-4o | dev500 | linguistic-lines-3way-k10 | 60.26 | 80.10 | 0.854 | 0.9967 | 0 |
| gpt-4o | dev500 | linguistic-lines-3way-k5 | 60.48 | 80.32 | 0.851 | 0.9967 | 0 |
| gpt-4o | dev500 | linguistic-marked-3way-k0 | 54.93 | 74.36 |  | 0.9978 | 0 |
| gpt-4o | dev500 | minimal-json-3way-k0 | 56.80 | 73.62 |  | 0.9890 | 0 |
| gpt-4o | dev500 | minimal-lines-3way-k0 | 57.82 | 78.00 | 0.848 | 0.9918 | 0 |
| gpt-4o | dev500 | minimal-marked-3way-k0 | 54.80 | 73.95 |  | 0.9995 | 0 |
| gpt-4o | dev500 | narrator-json-3way-k0 | 56.74 | 72.89 |  | 1.0000 | 0 |
| gpt-4o | dev500 | narrator-lines-3way-k0 | 53.95 | 74.51 | 0.832 | 0.9967 | 0 |
| gpt-4o | dev500 | narrator-marked-3way-k0 | 52.49 | 72.87 |  | 0.9992 | 0 |
| gpt-4o | test | linguistic-lines-3way-k5 | 59.44 | 79.67 | 0.841 | 0.9983 | 0 |
| deepseek-v4-pro | dev500 | linguistic-json-3way-k0 | 59.09 | 77.91 |  | 0.9951 | 0 |
| deepseek-v4-pro | dev500 | linguistic-lines-2way-k5 |  | 80.03 | 0.781 | 0.9951 | 0 |
| deepseek-v4-pro | dev500 | linguistic-lines-3way-k0 | 60.68 | 78.87 | 0.777 | 0.9918 | 0 |
| deepseek-v4-pro | dev500 | linguistic-lines-3way-k1 | 59.49 | 76.75 | 0.757 | 0.9918 | 0 |
| deepseek-v4-pro | dev500 | linguistic-lines-3way-k10 | 61.16 | 80.21 | 0.792 | 0.9918 | 0 |
| deepseek-v4-pro | dev500 | linguistic-lines-3way-k5 | 61.91 | 79.77 | 0.781 | 0.9918 | 0 |
| deepseek-v4-pro | dev500 | linguistic-marked-3way-k0 | 56.99 | 73.53 |  | 0.9884 | 0 |
| deepseek-v4-pro | dev500 | minimal-json-3way-k0 | 56.85 | 75.85 |  | 0.9890 | 0 |
| deepseek-v4-pro | dev500 | minimal-lines-3way-k0 | 60.31 | 79.52 | 0.791 | 0.9918 | 0 |
| deepseek-v4-pro | dev500 | minimal-marked-3way-k0 | 56.37 | 71.84 |  | 0.9765 | 0 |
| deepseek-v4-pro | dev500 | narrator-json-3way-k0 | 59.84 | 77.00 |  | 1.0000 | 0 |
| deepseek-v4-pro | dev500 | narrator-lines-3way-k0 | 59.19 | 78.18 | 0.773 | 0.9918 | 0 |
| deepseek-v4-pro | dev500 | narrator-marked-3way-k0 | 56.54 | 71.88 |  | 0.9729 | 0 |
| deepseek-v4-pro | test | linguistic-lines-3way-k5 | 60.52 | 79.21 | 0.780 | 0.9965 | 0 |
| mistral-large-3 | dev500 | linguistic-json-3way-k0 | 59.80 | 77.07 |  | 0.9951 | 0 |
| mistral-large-3 | dev500 | linguistic-lines-2way-k0 |  | 79.24 |  | 0.9951 | 0 |
| mistral-large-3 | dev500 | linguistic-lines-3way-k0 | 60.98 | 79.44 |  | 0.9918 | 0 |
| mistral-large-3 | dev500 | linguistic-lines-3way-k1 | 59.95 | 79.01 |  | 0.9918 | 0 |
| mistral-large-3 | dev500 | linguistic-lines-3way-k10 | 60.98 | 79.60 |  | 0.9918 | 0 |
| mistral-large-3 | dev500 | linguistic-lines-3way-k5 | 60.87 | 79.66 |  | 0.9918 | 0 |
| mistral-large-3 | dev500 | linguistic-marked-3way-k0 | 52.43 | 73.14 |  | 0.9667 | 0 |
| mistral-large-3 | dev500 | minimal-json-3way-k0 | 59.49 | 78.01 |  | 0.9890 | 0 |
| mistral-large-3 | dev500 | minimal-lines-3way-k0 | 60.32 | 79.79 |  | 0.9918 | 0 |
| mistral-large-3 | dev500 | minimal-marked-3way-k0 | 47.84 | 69.10 |  | 0.9659 | 0 |
| mistral-large-3 | dev500 | narrator-json-3way-k0 | 58.70 | 76.71 |  | 1.0000 | 0 |
| mistral-large-3 | dev500 | narrator-lines-3way-k0 | 58.52 | 78.20 |  | 0.9918 | 0 |
| mistral-large-3 | dev500 | narrator-marked-3way-k0 | 42.41 | 65.22 |  | 0.9559 | 0 |
| mistral-large-3 | test | linguistic-lines-3way-k0 | 59.73 | 78.93 |  | 0.9964 | 0 |
| gpt-5.5 | test200 | linguistic-lines-3way-k5 | 58.75 | 78.88 |  | 1.0000 | 0 |

Estimated spending so far: $73.00 (list prices; check the Azure cost view).
