import numpy as np
import pytest

from prosody_llms.data.corpus import read_split
from prosody_llms.probes import LabelSpace, RidgeAccumulator


def synthetic(n=400, dim=5, layers=2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    x = rng.normal(size=(layers, n, dim)) * np.array([1, 10, 0.1, 1, 1])  # mixed scales
    x[1, :, 0] += 3.0 * y  # layer 1 carries the label, layer 0 does not
    x[1, :, 2] += 0.3 * y
    return x, y


def test_lda_learns_ordered_classes_where_ridge_masks_the_middle_class():
    x, y = synthetic()  # the label lies along one direction: 0 < 1 < 2
    acc = RidgeAccumulator(2, 5, 3)
    acc.update(x, y)
    lda = acc.solve_lda(shrinkage=0.1).predict(x)
    ridge = acc.solve(alpha=1.0).predict(x)
    assert (lda[1] == y).mean() > 0.95
    assert (lda[0] == y).mean() < 0.5  # the uninformative layer stays at chance
    # Ridge onto one-hot labels under-predicts the middle class here ("masking").
    assert (ridge[1] == 1).sum() < (y == 1).sum() / 2
    with pytest.raises(ValueError):
        acc.solve_lda(shrinkage=2.0)


def test_ridge_learns_class_specific_directions():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=300)
    x = rng.normal(size=(1, 300, 4))
    x[0, np.arange(300), y] += 3.0  # one direction per class
    acc = RidgeAccumulator(1, 4, 3)
    acc.update(x, y)
    assert (acc.solve(alpha=1.0).predict(x)[0] == y).mean() > 0.9


def test_ridge_matches_direct_solution():
    x, y = synthetic(n=200)
    acc = RidgeAccumulator(2, 5, 3)
    acc.update(x, y)
    alpha = 3.0
    probe = acc.solve(alpha)
    # Direct ridge on standardised features with an unpenalised intercept.
    xl = x[1]
    mu, sd = xl.mean(0), xl.std(0)
    z = (xl - mu) / sd
    yy = np.eye(3)[y]
    w = np.linalg.solve(z.T @ z + alpha * np.eye(5), z.T @ (yy - yy.mean(0)))
    expected = z @ w + yy.mean(0)
    assert np.allclose(probe.scores(x)[1], expected, atol=1e-8)


def test_ridge_merge_state_and_unscored_words():
    x, y = synthetic(n=100)
    full = RidgeAccumulator(2, 5, 3)
    full.update(x, y)
    a, b = RidgeAccumulator(2, 5, 3), RidgeAccumulator(2, 5, 3)
    a.update(x[:, :60], y[:60])
    b.update(x[:, 60:], y[60:])
    a.merge(RidgeAccumulator.from_state(b.state()))
    assert np.allclose(a.solve(1.0).weights, full.solve(1.0).weights)
    y2 = y.copy()
    y2[:10] = -1  # punctuation is ignored
    c = RidgeAccumulator(2, 5, 3)
    c.update(x, y2)
    assert c.n == 90
    with pytest.raises(ValueError):
        RidgeAccumulator(1, 2, 2).solve(1.0)


def test_label_space(corpus_dir):
    u = read_split(corpus_dir / "train_360.txt")[0]  # The cat sat .  -> 0 2 1 NA
    assert list(LabelSpace(3).encode(u)) == [0, 2, 1, -1]
    assert list(LabelSpace(2).encode(u)) == [0, 1, 1, -1]
    assert LabelSpace(3).decode([0, 2, 1, 0], u) == ["0", "2", "1", "NA"]
