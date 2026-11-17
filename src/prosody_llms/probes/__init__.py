"""Probes on frozen word representations.

``ridge`` fits a linear probe for every layer from sufficient statistics accumulated in
one pass over the data, so features for large models never have to be stored. The
logistic, MLP and BiLSTM probes are in ``prosody_llms.probing.nonlinear``.
"""

from prosody_llms.probes.base import LabelSpace, Probe
from prosody_llms.probes.ridge import RidgeAccumulator, RidgeProbe

__all__ = ["LabelSpace", "Probe", "RidgeAccumulator", "RidgeProbe"]
