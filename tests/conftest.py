import os
from pathlib import Path

import pytest

TRAIN = """<file>\t100_1_000000_000000.txt
The\t0\t0\t0.000\t0.000
cat\t2\t2\t2.100\t1.500
sat\t1\t0\t0.900\t0.100
.\tNA\tNA\tNA\tNA
<file>\t100_1_000000_000001.txt
A\t0\t0\t0.050\t0.000
dog\t2\t1\t1.800\t0.700
runs\t0\t2\t0.200\t2.0
<file>\t200_2_000001_000000.txt
The\t0\t0\t0.010\t0.000
dog\t1\t0\t0.700\t0.200
sat\t2\t2\t1.900\t1.300
!\tNA\tNA\tNA\tNA
"""

DEV = """<file>\t300_3_000000_000000.txt
The\t0\t0\t0.000\t0.000
cats\t2\t2\t2.000\t1.000
<file>\t300_3_000000_000001.txt
A\t0\t0\t0.000\t0.000
dog\t1\t2\t0.800\t1.800
.\tNA\tNA\tNA\tNA
"""

TEST = """<file>\t400_4_000000_000000.txt
The\t0\t0\t0.000\t0.000
dog\t2\t1\t2.300\t0.600
sat\t1\t2\t0.700\t1.900
<file>\t400_4_000000_000001.txt
A\t0\t0\t0.000\t0.000
cat\t2\t0\t1.700\t0.300
runs\t0\t2\t0.100\t1.400
,\tNA\tNA\tNA\tNA
<file>\t500_5_000000_000000.txt
dogs\t2\t2\t2.200\t2.0
"""

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "cat", "sat", "a", "dog",
         "runs", "##s", ".", "!", ","]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "train_360.txt").write_text(TRAIN, encoding="utf-8")
    (d / "train_100.txt").write_text(TRAIN, encoding="utf-8")
    (d / "dev.txt").write_text(DEV, encoding="utf-8")
    (d / "test.txt").write_text(TEST, encoding="utf-8")
    return d


@pytest.fixture
def tiny_bert(tmp_path: Path) -> Path:
    """A randomly initialised 2-layer BERT with a 15-word vocabulary (no download)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    d = tmp_path / "tiny-bert"
    d.mkdir()
    # transformers >= 5 takes the vocabulary as `vocab` (a path or a dict).
    tok = transformers.BertTokenizer(vocab={w: i for i, w in enumerate(VOCAB)},
                                     do_lower_case=True)
    tok.save_pretrained(d)
    cfg = transformers.BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=2,
                                  num_attention_heads=2, intermediate_size=32,
                                  max_position_embeddings=64)
    transformers.BertModel(cfg).save_pretrained(d)
    return d


@pytest.fixture
def real_data_dir() -> Path:
    d = Path(os.environ.get("PROSODY_DATA", "data"))
    if not (d / "test.txt").is_file():
        pytest.skip("corpus not downloaded (run `make data`)")
    return d
