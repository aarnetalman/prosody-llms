"""Speech feature extractors with the same interface as the text extractor.

``align(utt)`` prepares an utterance (audio, word timings) and ``forward(batch)`` returns
one ``(layers, tokens, dim)`` tensor per utterance: for every corpus token the mean of
the model's frames whose centre lies inside the word's time span (the nearest frame if
none does). Tokens without a timing (punctuation, unaligned words) get zero vectors;
``word_mask(utt)`` says which tokens have a timing, so the probing runner can leave the
others out of training and scoring.

Utterances are run one at a time: no padding, so features do not depend on batching
(some of these models have no attention mask for padded audio).

* ``ssl`` (WavLM, HuBERT): every transformer layer, 20 ms frames.
* ``whisper``: every encoder layer; the input is Whisper's 30 s window, 20 ms frames;
  words after 30 s get no features.
* ``qwen2-audio``: the language model's hidden states at the audio token positions
  (``part="llm"``, 40 ms per token), or its audio encoder (``part="encoder"``, 20 ms).
  The prompt holds only the audio, no transcript.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prosody_llms.audio.base import LIBRITTS_SUBSETS, libritts_path
from prosody_llms.audio.timings import UtteranceTimings, load_timings
from prosody_llms.data.corpus import Utterance

SAMPLE_RATE = 16000


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the sample rate (16-bit PCM WAV)."""
    with wave.open(str(path), "rb") as w:
        n, sr, width, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: {8 * width}-bit audio; 16-bit PCM expected")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE, zeros: int = 16) -> np.ndarray:
    """Band-limited polyphase resampling (Kaiser-windowed sinc), e.g. 24 kHz -> 16 kHz."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    cutoff = 1.0 / max(up, down)
    half = zeros * max(up, down)
    t = np.arange(-half, half + 1)
    h = cutoff * np.sinc(cutoff * t) * np.kaiser(len(t), 8.6)
    h *= up / h.sum()
    y = np.zeros(len(x) * up, dtype=np.float64)
    y[::up] = x
    y = np.convolve(y, h, mode="same")[::down]
    return y.astype(np.float32)


@dataclass
class AlignedAudio:
    """An utterance ready for a speech model. The audio is read when the model needs it
    (``samples()``), so preparing a whole split costs only the WAV headers."""

    utt: Utterance
    path: Path
    timings: UtteranceTimings | None
    input_ids: range = field(default_factory=lambda: range(0))  # length only (batching)

    def samples(self) -> np.ndarray:
        x, sr = read_wav(self.path)
        return resample(x, sr)

    def spans(self) -> list[tuple[float, float] | None]:
        if self.timings is None:
            return [None] * len(self.utt.tokens)
        return [(t.start, t.end) if t.aligned else None for t in self.timings.timings]


def pool_frames(hidden, spans, frame_s: float, offset_s: float):
    """(layers, frames, dim) -> (layers, tokens, dim): mean of the frames whose centre
    (``offset_s + k * frame_s``) is inside each span; zeros for tokens without a span or
    beyond the last frame."""
    import torch

    n_layers, n_frames, dim = hidden.shape
    out = torch.zeros((n_layers, len(spans), dim), dtype=torch.float32, device=hidden.device)
    beyond = 0
    for i, sp in enumerate(spans):
        if sp is None:
            continue
        s, e = sp
        lo = max(0, math.ceil((s - offset_s) / frame_s - 1e-9))
        hi = min(n_frames - 1, math.floor((e - offset_s) / frame_s - 1e-9))
        if lo > hi:  # no centre inside: nearest frame
            lo = hi = min(n_frames - 1, max(0, round(((s + e) / 2 - offset_s) / frame_s)))
        if s >= n_frames * frame_s + offset_s:
            beyond += 1
            continue
        out[:, i] = hidden[:, lo:hi + 1].float().mean(dim=1)
    return out, beyond


class AudioExtractor:
    """Base: loading audio and timings, masks, batching; subclasses define ``_hidden``."""

    frame_s = 0.02
    offset_s = 0.01
    max_seconds: float | None = None  # input window (Whisper, Qwen2-Audio: 30 s)

    def __init__(self, audio_dir: Path, name: str):
        self.audio_dir = Path(audio_dir)
        self.name = name
        self.split_of: dict[str, str] = {}
        self._timings: dict[str, UtteranceTimings | None] = {}
        self.alignment_stats = {"utterances": 0, "tokens": 0, "aligned_tokens": 0, "beyond_window": 0}

    # -- registration and masks
    def register(self, splits: dict[str, Sequence[Utterance]]) -> None:
        from prosody_llms.audio.download import textgrid_root

        root = textgrid_root(self.audio_dir)
        for split, utts in splits.items():
            for u in utts:
                self.split_of[u.utt_id] = split
                self._timings[u.utt_id] = load_timings(root, LIBRITTS_SUBSETS[split], u)

    def word_mask(self, u: Utterance) -> list[bool]:
        t = self._timings.get(u.utt_id)
        if t is None:
            return [False] * len(u.tokens)
        return [m and not tok.is_punct and (self.max_seconds is None or w.start < self.max_seconds)
                for m, tok, w in zip(t.mask(), u.tokens, t.timings)]

    def align(self, u: Utterance) -> AlignedAudio:
        path = libritts_path(self.audio_dir, self.split_of[u.utt_id], u.utt_id)
        with wave.open(str(path), "rb") as w:
            seconds = w.getnframes() / w.getframerate()
        return AlignedAudio(u, path, self._timings.get(u.utt_id), range(int(seconds * 50) + 1))

    # -- forward
    def forward(self, aligned: Sequence[AlignedAudio]):
        import torch

        out = []
        with torch.no_grad():
            for a in aligned:
                hidden, frame_s, offset_s = self._hidden(a)
                mask = self.word_mask(a.utt)
                spans = [sp if m else None for sp, m in zip(a.spans(), mask)]
                pooled, beyond = pool_frames(hidden, spans, frame_s, offset_s)
                self.alignment_stats["utterances"] += 1
                self.alignment_stats["tokens"] += len(mask)
                self.alignment_stats["aligned_tokens"] += sum(mask)
                self.alignment_stats["beyond_window"] += beyond
                out.append(pooled)
        return out

    def _hidden(self, a: AlignedAudio):
        """(layers, frames, dim) tensor, frame length and centre of frame 0 (seconds)."""
        raise NotImplementedError

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def dim(self) -> int:
        return self._dim


def _load(cls, model_id: str, revision: str | None, dtype: str, device: str | None):
    """``from_pretrained`` that refuses checkpoints with missing weights."""
    import torch

    model, info = cls.from_pretrained(model_id, revision=revision, torch_dtype=getattr(torch, dtype),
                                      output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "masked_spec_embed" not in k]
    if missing:
        raise RuntimeError(f"{model_id}: {len(missing)} weights missing, e.g. {missing[:5]}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


class SSLExtractor(AudioExtractor):
    """WavLM, HuBERT, wav2vec 2.0: all hidden states (layer 0 = after the CNN and
    projection)."""

    def __init__(self, model_id, revision, audio_dir, name, dtype="float32", device=None):
        super().__init__(audio_dir, name)
        from transformers import AutoFeatureExtractor, AutoModel

        self.fe = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        self.model = _load(AutoModel, model_id, revision, dtype, device)
        self.device = next(self.model.parameters()).device
        cfg = self.model.config
        self._n_layers, self._dim = cfg.num_hidden_layers + 1, cfg.hidden_size
        self.model_class, self.key_mapping = type(self.model).__name__, None
        stride = int(np.prod(cfg.conv_stride))
        self.frame_s = stride / SAMPLE_RATE
        rf = 1
        for k, s in zip(reversed(cfg.conv_kernel), reversed(cfg.conv_stride)):
            rf = (rf - 1) * s + k
        self.offset_s = rf / 2 / SAMPLE_RATE

    def _hidden(self, a):
        import torch

        inp = self.fe(a.samples(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = inp["input_values"].to(self.device, dtype=next(self.model.parameters()).dtype)
        out = self.model(input_values=x, output_hidden_states=True)
        return torch.stack([h[0] for h in out.hidden_states]), self.frame_s, self.offset_s


class WhisperEncoderExtractor(AudioExtractor):
    """Whisper's encoder: all hidden states over the 30 s window (layer 0 = after the
    convolutions and positions)."""

    def __init__(self, model_id, revision, audio_dir, name, dtype="float32", device=None):
        super().__init__(audio_dir, name)
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.fe = WhisperFeatureExtractor.from_pretrained(model_id, revision=revision)
        model = _load(WhisperModel, model_id, revision, dtype, device)
        self.encoder = model.get_encoder()
        del model.decoder
        self.device = next(self.encoder.parameters()).device
        cfg = self.encoder.config
        self._n_layers, self._dim = cfg.encoder_layers + 1, cfg.d_model
        self.model_class, self.key_mapping = "WhisperEncoder", None
        self.max_seconds = 30.0

    def _hidden(self, a):
        import torch

        inp = self.fe(a.samples(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = inp["input_features"].to(self.device, dtype=next(self.encoder.parameters()).dtype)
        out = self.encoder(input_features=x, output_hidden_states=True)
        return torch.stack([h[0] for h in out.hidden_states]), 0.02, 0.01


class Qwen2AudioExtractor(AudioExtractor):
    """Qwen2-Audio: the language model at the audio positions (``part="llm"``) or the
    audio encoder (``part="encoder"``)."""

    PROMPT = "<|audio_bos|><|AUDIO|><|audio_eos|>"

    def __init__(self, model_id, revision, audio_dir, name, dtype="bfloat16", device=None, part="llm"):
        super().__init__(audio_dir, name)
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = _load(Qwen2AudioForConditionalGeneration, model_id, revision, dtype, device)
        self.device = next(self.model.parameters()).device
        self.part = part
        self.max_seconds = 30.0
        cfg = self.model.config
        if part == "llm":
            self._n_layers = cfg.text_config.num_hidden_layers + 1
            self._dim = cfg.text_config.hidden_size
        else:
            self._n_layers = cfg.audio_config.encoder_layers + 1
            self._dim = cfg.audio_config.d_model
        self.audio_token_id = getattr(cfg, "audio_token_id", None) or getattr(cfg, "audio_token_index")
        self.model_class, self.key_mapping = f"{type(self.model).__name__}/{part}", None

    def _hidden(self, a):
        import torch

        inp = self.processor(text=self.PROMPT, audio=a.samples(), sampling_rate=SAMPLE_RATE,
                             return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        if "input_features" in inp:
            inp["input_features"] = inp["input_features"].to(next(self.model.parameters()).dtype)
        if self.part == "encoder":
            tower = getattr(self.model, "audio_tower", None) or self.model.model.audio_tower
            out = tower(inp["input_features"], output_hidden_states=True)
            hs = [h[0] for h in out.hidden_states]
            n = max(h.shape[0] for h in hs)
            # The last state comes after the encoder's 2x average pooling (40 ms frames):
            # repeat each frame to put it on the 20 ms grid of the others.
            hs = [h if h.shape[0] == n else h.repeat_interleave(n // h.shape[0], dim=0) for h in hs]
            return torch.stack(hs), 0.02, 0.01
        out = self.model(**inp, output_hidden_states=True)
        pos = (inp["input_ids"][0] == self.audio_token_id).nonzero().flatten()
        hs = torch.stack([h[0, pos] for h in out.hidden_states])
        return hs, 0.04, 0.02


def make_audio_extractor(spec, audio_dir: Path, dtype: str, device: str | None, part: str = "llm"):
    """The extractor for a registry entry of kind ``speech`` or ``speech_text``."""
    if spec.kind == "speech_text":
        return Qwen2AudioExtractor(spec.id, spec.revision, audio_dir, spec.name,
                                   dtype="bfloat16" if dtype == "float32" else dtype, device=device,
                                   part=part)
    if "whisper" in spec.id.lower():
        return WhisperEncoderExtractor(spec.id, spec.revision, audio_dir, spec.name, dtype, device)
    return SSLExtractor(spec.id, spec.revision, audio_dir, spec.name, dtype, device)
