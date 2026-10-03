"""Unit and integration tests for Multilingual Voice Support (Stage 11)."""
import io
import wave
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.voice.language import (
    LANG_AUTO,
    LANG_EN,
    LANG_HI,
    LANG_ML,
    detect_language,
    get_language_prompt_instruction,
    resolve_language,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    """Generates a small valid WAV file for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)  # 50ms silence
    return buf.getvalue()


def test_language_detection():
    # English
    assert detect_language("Who are the doctors in Cardiology?") == LANG_EN
    assert detect_language("Tell me hospital visiting hours") == LANG_EN

    # Malayalam (U+0D00 to U+0D7F)
    assert detect_language("കാർഡിയോളജി വിഭാഗത്തിൽ ഏതൊക്കെ ഡോക്ടർമാരുണ്ട്?") == LANG_ML
    assert detect_language("നമസ്കാരം, കുട്ടികളുടെ ഡോക്ടർ ആരാണ്?") == LANG_ML

    # Hindi (Devanagari U+0900 to U+097F)
    assert detect_language("कार्डियोलॉजी विभाग में कौन से डॉक्टर उपलब्ध हैं?") == LANG_HI
    assert detect_language("अस्पताल का समय क्या है?") == LANG_HI


def test_resolve_language():
    assert resolve_language("en-IN") == LANG_EN
    assert resolve_language("ml-IN") == LANG_ML
    assert resolve_language("hi-IN") == LANG_HI
    assert resolve_language("auto", "കാർഡിയോളജി") == LANG_ML
    assert resolve_language("auto", "कार्डियोलॉजी") == LANG_HI
    assert resolve_language("auto", "Cardiology") == LANG_EN
    assert resolve_language(None, "General Medicine") == LANG_EN


def test_language_prompt_instructions():
    ml_instr = get_language_prompt_instruction(LANG_ML)
    assert "Malayalam" in ml_instr or "മലയാളം" in ml_instr

    hi_instr = get_language_prompt_instruction(LANG_HI)
    assert "Hindi" in hi_instr or "हिन्दी" in hi_instr

    en_instr = get_language_prompt_instruction(LANG_EN)
    assert "English" in en_instr


def test_voice_transcribe_empty_payload(client):
    r = client.post("/api/voice/transcribe", files={"audio": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400
    assert "Empty audio payload" in r.text or "too short" in r.text


def test_voice_synthesize_empty_payload(client):
    r = client.post("/api/voice/synthesize", json={"text": "", "language": "en-IN"})
    assert r.status_code in (400, 422)


def test_chat_endpoint_accepts_language(client):
    payload = {
        "message": "Where is the hospital located?",
        "language": "en-IN",
        "history": [],
    }
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
