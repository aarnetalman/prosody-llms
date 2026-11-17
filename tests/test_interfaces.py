import json
from pathlib import Path

from prosody_llms.audio import WordTiming, libritts_path
from prosody_llms.prompting.cache import request_key


def test_request_key_is_stable_and_sensitive():
    body = {"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "Mark."}]}
    k = request_key("azure-openai", body)
    assert k == request_key("azure-openai", json.loads(json.dumps(body)))
    assert k != request_key("azure-openai", {**body, "temperature": 0.1})
    assert k != request_key("azure-inference", body)


def test_libritts_path_and_timing():
    p = libritts_path("/audio", "test", "121_127105_000000_000000")
    assert p == Path("/audio/LibriTTS/test-clean/121/127105/121_127105_000000_000000.wav")
    assert WordTiming("a", 0.1, 0.3).aligned
    assert not WordTiming("a", None, None).aligned
    assert not WordTiming("a", 0.3, 0.3).aligned
