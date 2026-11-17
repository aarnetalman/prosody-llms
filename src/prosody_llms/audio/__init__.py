"""Audio side: LibriTTS audio, word timings and speech features, label
reconstruction, audio prompting and the text-against-audio comparison.

Speech extractors return the same ``(layers, words, dim)`` arrays as text extractors,
so the probes and evaluation are shared between text and audio.
"""

from prosody_llms.audio.base import AudioFeatureExtractor, WordTiming, libritts_path

__all__ = ["AudioFeatureExtractor", "WordTiming", "libritts_path"]
