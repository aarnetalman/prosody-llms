# prosody-llms

Code and results for the paper

> Aarne Talman and Stergios Chatzikyriakidis. 2026.
> **Language models represent more about emphasis than they can tell you.** arXiv preprint.

The paper asks how much of **prosodic prominence** (which words a reader stresses) language
models recover from text, whether they can report it when asked, and what the speech
signal adds. Everything behind it is here: the code, the results of every run, and the
commands that reproduce them.

## Quick start

| I want to... | Do this | Needs |
|---|---|---|
| regenerate the paper's tables, numbers and figure data from the committed results | `make setup data analysis` | CPU, internet for the corpus; a few minutes |
| check the installation | `make test` | CPU, about a minute |
| rerun one experiment | `python scripts/reproduce.py groups`, then `run <group>` ([below](#reproducing-the-experiments)) | depends on the group, mostly a GPU |
| reuse a part (word-token alignment, probes, prompt parser, 2019 port) | see [Layout](#layout) | |

## What the repository covers

- **Fine-tuning** BERT-style encoders end to end, including a faithful reproduction of
  Talman et al. (2019).
- **Probing** frozen encoders and decoder LLMs (0.1B to 32B parameters):
  - linear probes on every layer;
  - logistic, MLP and BiLSTM classifiers on the best layer;
  - controls on the embedding layer and on random word vectors.
- **Prompting** hosted LLMs (GPT-4o, DeepSeek-V4-Pro, Mistral Large 3, GPT-5.5) and an open
  LLM with up to 500 in-context examples.
- **Speech models**:
  - probes on WavLM, HuBERT, Whisper and Qwen2-Audio;
  - Qwen2-Audio-Instruct prompted with and without the recording;
  - the labels reconstructed from the audio with the labelling tool itself.
- **Analysis**: every table, figure and number in the paper, from the committed `results/`.

### From the paper to the code

| Paper | Experiment group | Results | Code |
|---|---|---|---|
| Section 3, Tables 1 and 11: data, label reconstruction, agreement between readers | `data`, `audio-reference` | `results/audio/` | `data/`, `audio/reconstruct.py` |
| Tables 2 and 3: models, main results | all groups | `results/*/` | `analysis/` |
| Section 5.1, Figures 1 to 3: text models, learning curves, agreement between systems, layers | `baselines`, `repro-2019`, `finetune`, `probe-linear`, `probe-strong`, `controls` | `results/repro_2019/`, `finetune/`, `probing*/` | `legacy/`, `probes/`, `probing/` |
| Section 5.2, Tables 4 to 7, Figures 4 and 5: probing against prompting | `prompting`, `icl`, `probe-strong` | `results/prompting/`, `icl/`, `probing_nonlinear/` | `prompting/`, `probing/` |
| Section 5.3, Table 8: speech models | `audio-linear`, `audio-strong`, `audio-prompting`, `compare` | `results/audio/`, `audio_prompting/` | `audio/` |
| Section 6, Tables 9 and 10: word types, error analysis | `analysis` | `results/analysis/` | `analysis/` |

Code paths are under `src/prosody_llms/`. `make analysis` writes every table of the paper to
`results/analysis/tables/`, the data behind every figure to `results/analysis/figures/`, and
every number quoted in the text, by name, to `results/analysis/numbers.csv`. These outputs
are committed, so they can be read without running anything.

## Installation

Python 3.11 or newer. The results were produced with Python 3.12, PyTorch 2.13,
Transformers 5.22.2 and NumPy 2.5.2 (`requirements.txt`). Every result file records the
versions and the device it was run with.

```bash
git clone https://github.com/aarnetalman/prosody-llms.git && cd prosody-llms
python -m venv .venv && source .venv/bin/activate
make setup                 # pip install -r requirements.txt && pip install -e .
make test                  # offline, CPU, about a minute
```

Install PyTorch for your platform first if you need a particular CUDA build.

## Data

The corpus of Talman et al. (2019) is word-level prominence labels for the LibriTTS
audiobooks (CC BY 4.0). It is downloaded from
[Helsinki-NLP/prosody](https://github.com/Helsinki-NLP/prosody) at a pinned commit and
checked against SHA-256 checksums. It is not stored here.

```bash
export PROSODY_DATA=$PWD/data      # default: data
export PROSODY_RUNS=$PWD/runs      # outputs; default: runs
make data
```

- **Splits:** `train_360` (training), `dev` and `test`, plus `train_100` for the audio
  experiments.
- **Tasks:** 3-way (labels 0, 1, 2) and 2-way (1 and 2 merged).
- **Scoring:**
  - Punctuation is never scored.
  - The metric is word-level accuracy, as in 2019; macro-F1, bootstrap intervals and McNemar
    tests are also reported.
  - The 2019 loader dropped the last utterance of every file. Comparisons with the 2019
    paper therefore use that subset (`test (2019 subset)`); everything else uses the full
    test set.

The audio experiments also need:
- the LibriTTS recordings, from OpenSLR;
- the public forced alignments of LibriTTS
  ([kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel));
- the labelling tool, the [Wavelet Prosody Toolkit](https://github.com/asuni/wavelet_prosody_toolkit).

The `audio-prep` group below fetches all three, at pinned versions. The audio goes to
`$PROSODY_AUDIO`, which defaults to `$PROSODY_DATA/../audio`.

## Models

`configs/models.toml` lists every model with the exact Hugging Face revision that was
used; nothing is loaded by name alone.

- **Downloading:** `python -m prosody_llms.prefetch` downloads the models into the
  Hugging Face cache (`HF_HOME`) and checks the downloads, so the jobs can then run offline
  (`HF_HUB_OFFLINE=1`).
- **Disk:** all models together take about 300 GB.
- **Memory:** the largest (30B and 32B) need one GPU with about 95 GB of memory in bfloat16.
- **Compute:** in total the runs took about 60 GPU hours on NVIDIA GH200 GPUs, the longest
  single run 8 hours.

## Reproducing the experiments

`scripts/reproduce.py` holds every command behind the paper, grouped by experiment. The
commands are plain `python -m prosody_llms...` calls that read only the environment
variables above, so they run the same on a workstation, a cloud machine or a cluster.

```bash
python scripts/reproduce.py groups              # the groups in order, with what they need
python scripts/reproduce.py list probe-linear   # the commands of one group
python scripts/reproduce.py run baselines       # run a group here, one command after another
```

| Group | Needs | What it does |
|---|---|---|
| `data` | internet | download and verify the corpus |
| `models` | internet, ~300 GB | download all models at their pinned revisions |
| `baselines` | CPU | majority, per-word and random baselines |
| `repro-2019` | GPU | the 2019 BERT-base recipe: 2 tasks × 5 seeds, and with 10% of the data |
| `finetune` | GPU | BERT-large, ModernBERT-large, XLM-R large: 2 tasks × 3 seeds |
| `probe-linear` | GPU | linear probes on every layer of 15 text models; learning-curve and discourse-context runs |
| `probe-strong` | GPU | logistic, MLP and BiLSTM on each model's best layer |
| `controls` | GPU | BiLSTM on the embedding layer and on random word vectors |
| `icl` | GPU | Qwen3.8 27B with 0–500 in-context examples |
| `prompting` | API key, paid | the hosted models (below) |
| `audio-prep` | internet | labelling toolkit, LibriTTS audio and alignments |
| `audio-reference` | CPU | duration-only baseline; labels reconstructed from audio |
| `audio-linear`, `audio-strong` | GPU | probes on the speech models and the text controls trained on `train_100` |
| `audio-prompting` | GPU | Qwen2-Audio-Instruct with and without the recording |
| `compare` | CPU | every system on identical test words |
| `collect` | CPU | copy the small result files into `results/` |
| `analysis` | CPU | the paper's tables and numbers |

The groups have to run in this order:
- the stronger probes and the controls read the best layer chosen by the linear probes;
- the comparison reads everything before it.

Within a group the commands are independent and can run in parallel. Every command can be
rerun safely: finished runs are skipped, and interrupted probing runs resume from their
last checkpoint.

### On a cluster

On a Slurm cluster, run each group as one array job, one command per task. Set the
account, partition and GPU type of your cluster in `slurm/array.sbatch` or on the command
line:

```bash
mkdir -p logs
python scripts/reproduce.py list probe-linear > jobs-probe-linear.txt
sbatch --array=1-$(wc -l < jobs-probe-linear.txt) --account=<account> --partition=<partition> \
  slurm/array.sbatch jobs-probe-linear.txt
```

Download the corpus, the audio and the models first (the groups `data`, `models` and
`audio-prep`) on a machine with internet, since compute nodes often have none. With other
schedulers, run each line of the job file as a job.

### Prompting the hosted models

The paper used Azure AI Foundry deployments of each model; any OpenAI-compatible endpoint
can be configured in `configs/prompting.toml`. Endpoints and keys come from the
environment and are never written to disk:

```bash
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_AI_INFERENCE_ENDPOINT=https://<resource>.services.ai.azure.com/models
export AZURE_AI_API_KEY=...
python -m prosody_llms.prompting.sweep                                  # plan and cost estimate only
python -m prosody_llms.prompting.sweep --confirm-spend --max-usd 100    # the paid run
```

**Nothing is spent by default.** Without `--confirm-spend` the sweep only prints the
number of requests and a cost estimate. With it:
- it stops at `--max-usd`, and at the total cap in `configs/prompting.toml`;
- every request and raw response is cached, so a rerun never pays twice.

The paper's sweep cost about USD 73 at list prices. It works in stages:
1. instruction variants × answer formats;
2. 1, 5 and 10 examples;
3. native 2-way prompts.

Settings are chosen on 500 development utterances, then the best setting of each model is
run on the test set (GPT-5.5 on 200 test utterances).

### After the runs

```bash
python scripts/collect_results.py     # copy small result files from $PROSODY_RUNS to results/
python -m prosody_llms.analysis      # tables and numbers
```

## Regenerating the paper's numbers

```bash
make analysis     # python -m prosody_llms.analysis
```

This reads only `results/` and the corpus, takes a few seconds on a CPU, and writes
`results/analysis/`:

- `all_results.csv`: every result as one table (system, task, test and dev scores);
- `main_table.md`: the main table, readable on GitHub;
- `tables/`: each table of the paper as CSV (`data`, `models`, `main_table`, `paired`,
  `behaviour`, `confusion`, `lexical`, `readers`, `two_readers`, `comparison`), and
  `breakdown.csv` and `overlap.csv` one level up;
- `figures/`: the data behind each figure (layer curves, learning curves, model size,
  number of examples, agreement between systems);
- `numbers.csv`: every number quoted in the text, by name;
- `error_analysis.md` and `visuals.json`: the error analysis and example sentences.

## Layout

```
src/prosody_llms/
  data/        corpus download (pinned, checksummed), reader, analysis subsets
  eval/        accuracy, macro-F1, per-class scores, bootstrap, McNemar; result formats
  features/    word-to-token alignment for any tokenizer, pooling, feature extractors
  probes/      one-pass ridge and LDA probes for all layers
  probing/     linear probing harness; logistic, MLP and BiLSTM classifiers
  prompting/   prompts, answer parser, API client, cache and budget; many-shot open models
  audio/       audio download, word timings, label reconstruction, duration baseline,
               speech extractors, audio prompting, the text-against-audio comparison
  legacy/      faithful port of the 2019 BERT code and baselines
  analysis/    tables, numbers and figure data for the paper; error analysis
  prefetch.py  download and verify the pinned models
configs/       models.toml (pinned models), prompting.toml (hosted models, prices)
scripts/       reproduce.py, collect_results.py, install_toolkit.sh
slurm/         array.sbatch: a generic array-job template
results/       small result files of every run behind the paper
tests/         pytest suite (offline; uses tiny randomly initialised models)
```

### The 2019 reproduction

`prosody_llms.legacy` ports the 2019 code to current libraries and keeps the original
behaviour that affects results:

- per-word tokenisation and first-piece labels;
- punctuation as a training class;
- no attention mask;
- Adam at 5e-5, batch 32, two epochs, and the final-epoch model.

There are two deviations. The label order is fixed (sorted), where in 2019 it depended on
Python's hash randomisation, and the kernels are deterministic. The reproduction reaches
83.0 / 68.5 (2-way / 3-way) against 83.2 / 68.6 in the 2019 paper.

## Citation

If you use the code or the results, please cite the paper, and the 2019 paper whose corpus
and task this builds on. Both are in `CITATION.cff`, which GitHub also offers in other
formats under "Cite this repository".

- Aarne Talman and Stergios Chatzikyriakidis. 2026. Language models represent more about
  emphasis than they can tell you. arXiv preprint.
- Aarne Talman, Antti Suni, Hande Celikkanat, Sofoklis Kakouros, Jörg Tiedemann and Martti
  Vainio. 2019. Predicting Prosodic Prominence from Text with Pre-trained Contextualized
  Word Representations. In *Proceedings of the 22nd Nordic Conference on Computational
  Linguistics (NoDaLiDa)*, pages 281–290.

## Licence

The code is under the MIT licence. The corpus and LibriTTS are under CC BY 4.0. The Wavelet
Prosody Toolkit and the models have their own licences.
