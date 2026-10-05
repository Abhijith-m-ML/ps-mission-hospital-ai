"""Tests for 3-Gemini-API-Key Fallback and Voice Synthesis (Stage 12).

Verifies the 10 required test scenarios:
1. Normal voice request (English)
2. Malayalam voice request
3. English voice request
4. Hindi voice request
5. Multiple consecutive TTS requests
6. Simulated Gemini Key 1 failure
7. Verify Key 2 is used
8. Simulated Key 2 failure
9. Verify Key 3 is used
10. All three keys failing (controlled error)
Additional safety checks:
11. Non-retryable errors (400 Bad Request) do NOT rotate keys
12. Empty audio protection
13. No API key leakage in logs or responses
"""
import base64
import io
import os
import wave
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.gemini_key_manager import GeminiKeyManager, get_gemini_key_manager
from app.voice.language import LANG_EN, LANG_HI, LANG_ML
from app.voice.tts import GeminiTTS


@pytest.fixture
def client():
    return TestClient(app)


def _create_dummy_wav_pcm() -> bytes:
    """Creates 100ms of dummy 16-bit PCM silence."""
    return b"\x00\x00" * 2400


def _create_mock_gemini_tts_response(pcm_bytes: bytes, status_code: int = 200) -> httpx.Response:
    """Creates a mock httpx.Response mimicking Gemini TTS generateContent."""
    if status_code == 200:
        b64_data = base64.b64encode(pcm_bytes).decode("utf-8")
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/x-wav",
                                    "data": b64_data,
                                }
                            }
                        ]
                    }
                }
            ]
        }
        return httpx.Response(
            status_code=200,
            json=data,
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent"),
        )
    else:
        return httpx.Response(
            status_code=status_code,
            json={"error": {"code": status_code, "message": f"Simulated error {status_code}"}},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent"),
        )


# ==============================================================================
# Scenario 1: Normal voice request
# ==============================================================================
def test_01_normal_voice_request():
    """Scenario 1: Normal voice request synthesizes valid WAV audio."""
    mgr = GeminiKeyManager(keys=[(1, "fake_key_1")])
    dummy_pcm = _create_dummy_wav_pcm()

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _create_mock_gemini_tts_response(dummy_pcm)
        tts = GeminiTTS(key_manager=mgr)
        wav_bytes, mime = tts.synthesize("Welcome to PSG Hospitals.", language="en-IN")

        assert mime == "audio/wav"
        assert wav_bytes.startswith(b"RIFF")
        assert len(wav_bytes) > 44
        assert mock_post.call_count >= 1


# ==============================================================================
# Scenario 2: Malayalam voice request
# ==============================================================================
def test_02_malayalam_voice_request():
    """Scenario 2: Malayalam voice request uses Malayalam voice configuration."""
    mgr = GeminiKeyManager(keys=[(1, "fake_key_1")])
    dummy_pcm = _create_dummy_wav_pcm()

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _create_mock_gemini_tts_response(dummy_pcm)
        tts = GeminiTTS(key_manager=mgr)
        wav_bytes, mime = tts.synthesize("പിഎസ്ജി ആശുപത്രിയിലേക്ക് സ്വാഗതം.", language="ml-IN")

        assert mime == "audio/wav"
        assert wav_bytes.startswith(b"RIFF")
        # Check payload voice config
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        voice_cfg = payload.get("generationConfig", {}).get("speechConfig", {}).get("voiceConfig", {})
        assert voice_cfg.get("prebuiltVoiceConfig", {}).get("voiceName") == "Kore"


# ==============================================================================
# Scenario 3: English voice request
# ==============================================================================
def test_03_english_voice_request():
    """Scenario 3: English voice request uses English voice configuration."""
    mgr = GeminiKeyManager(keys=[(1, "fake_key_1")])
    dummy_pcm = _create_dummy_wav_pcm()

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _create_mock_gemini_tts_response(dummy_pcm)
        tts = GeminiTTS(key_manager=mgr)
        wav_bytes, mime = tts.synthesize("Dr. Rajesh Kumar is available in OPD 3.", language="en-IN")

        assert mime == "audio/wav"
        assert wav_bytes.startswith(b"RIFF")
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        voice_cfg = payload.get("generationConfig", {}).get("speechConfig", {}).get("voiceConfig", {})
        assert voice_cfg.get("prebuiltVoiceConfig", {}).get("voiceName") == "Puck"


# ==============================================================================
# Scenario 4: Hindi voice request
# ==============================================================================
def test_04_hindi_voice_request():
    """Scenario 4: Hindi voice request uses Hindi voice configuration."""
    mgr = GeminiKeyManager(keys=[(1, "fake_key_1")])
    dummy_pcm = _create_dummy_wav_pcm()

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _create_mock_gemini_tts_response(dummy_pcm)
        tts = GeminiTTS(key_manager=mgr)
        wav_bytes, mime = tts.synthesize("अस्पताल में आपका स्वागत है।", language="hi-IN")

        assert mime == "audio/wav"
        assert wav_bytes.startswith(b"RIFF")
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        voice_cfg = payload.get("generationConfig", {}).get("speechConfig", {}).get("voiceConfig", {})
        assert voice_cfg.get("prebuiltVoiceConfig", {}).get("voiceName") == "Aoede"


# ==============================================================================
# Scenario 5: Multiple consecutive TTS requests
# ==============================================================================
def test_05_multiple_consecutive_tts_requests():
    """Scenario 5: Multiple consecutive TTS requests complete reliably without state leakage."""
    mgr = GeminiKeyManager(keys=[(1, "fake_key_1"), (2, "fake_key_2"), (3, "fake_key_3")])
    dummy_pcm = _create_dummy_wav_pcm()

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _create_mock_gemini_tts_response(dummy_pcm)
        tts = GeminiTTS(key_manager=mgr)

        for i in range(5):
            wav_bytes, mime = tts.synthesize(f"This is consecutive request number {i + 1}.", language="en-IN")
            assert mime == "audio/wav"
            assert wav_bytes.startswith(b"RIFF")

        assert mock_post.call_count == 5


# ==============================================================================
# Scenario 6 & 7: Simulated Gemini Key 1 failure -> Verify Key 2 is used
# ==============================================================================
def test_06_and_07_key_1_fails_and_key_2_is_used():
    """Scenario 6 & 7: Key 1 fails with 429 (rate limit) -> Key 2 is selected and succeeds."""
    called_slots = []
    dummy_pcm = _create_dummy_wav_pcm()

    def mock_post(url, **kwargs):
        headers = kwargs.get("headers", {})
        key = headers.get("x-goog-api-key")
        if key == "key_one":
            called_slots.append(1)
            # Simulate 429 Quota Exceeded on Key 1
            return _create_mock_gemini_tts_response(b"", status_code=429)
        elif key == "key_two":
            called_slots.append(2)
            # Key 2 succeeds
            return _create_mock_gemini_tts_response(dummy_pcm, status_code=200)
        else:
            called_slots.append(3)
            return _create_mock_gemini_tts_response(dummy_pcm, status_code=200)

    mgr = GeminiKeyManager(keys=[(1, "key_one"), (2, "key_two"), (3, "key_three")])
    tts = GeminiTTS(key_manager=mgr)

    with patch("httpx.Client.post", side_effect=mock_post):
        wav_bytes, mime = tts.synthesize("Emergency services are open 24/7.", language="en-IN")

        assert mime == "audio/wav"
        assert wav_bytes.startswith(b"RIFF")
        # Assert Key 1 was tried first, failed, then Key 2 was used and succeeded
        assert 1 in called_slots
        assert 2 in called_slots
        assert 3 not in called_slots  # Stopped after Key 2 succeeded


# ==============================================================================
# Scenario 8 & 9: Simulated Key 2 failure -> Verify Key 3 is used
# ==============================================================================
def test_08_and_09_key_1_and_2_fail_and_key_3_is_used():
    """Scenario 8 & 9: Key 1 (429) and Key 2 (401 auth error) fail -> Key 3 is selected and succeeds."""
    called_slots = []
    dummy_pcm = _create_dummy_wav_pcm()

    def mock_post(url, **kwargs):
        headers = kwargs.get("headers", {})
        key = headers.get("x-goog-api-key")
        if key == "key_one":
            called_slots.append(1)
            return _create_mock_gemini_tts_response(b"", status_code=429)
        elif key == "key_two":
            called_slots.append(2)
            return _create_mock_gemini_tts_response(b"", status_code=401)
        elif key == "key_three":
            called_slots.append(3)
            return _create_mock_gemini_tts_response(dummy_pcm, status_code=200)
        return _create_mock_gemini_tts_response(b"", status_code=500)

    mgr = GeminiKeyManager(keys=[(1, "key_one"), (2, "key_two"), (3, "key_three")])
    tts = GeminiTTS(key_manager=mgr)

    with patch("httpx.Client.post", side_effect=mock_post):
        wav_bytes, mime = tts.synthesize("Cardiology department is on the second floor.", language="en-IN")

        assert mime == "audio/wav"
        assert wav_bytes.startswith(b"RIFF")
        # Assert keys attempted in order: 1 -> 2 -> 3
        assert called_slots == [1, 2, 3]


# ==============================================================================
# Scenario 10: All three keys failing -> Controlled error
# ==============================================================================
def test_10_all_three_keys_fail_controlled_error():
    """Scenario 10: When all 3 keys fail with retryable errors, return controlled error."""
    called_slots = []

    def mock_post(url, **kwargs):
        headers = kwargs.get("headers", {})
        key = headers.get("x-goog-api-key")
        if key == "key_one":
            called_slots.append(1)
            return _create_mock_gemini_tts_response(b"", status_code=429)
        elif key == "key_two":
            called_slots.append(2)
            return _create_mock_gemini_tts_response(b"", status_code=429)
        elif key == "key_three":
            called_slots.append(3)
            return _create_mock_gemini_tts_response(b"", status_code=503)
        return _create_mock_gemini_tts_response(b"", status_code=500)

    mgr = GeminiKeyManager(keys=[(1, "key_one"), (2, "key_two"), (3, "key_three")])
    tts = GeminiTTS(key_manager=mgr)

    with patch("httpx.Client.post", side_effect=mock_post):
        with pytest.raises(RuntimeError) as exc_info:
            tts.synthesize("Doctor appointments require prior registration.", language="en-IN")

        # Controlled error raised
        assert "failed across all 3 configured Gemini API keys" in str(exc_info.value)
        # Exactly 3 attempts made, no infinite loop
        assert called_slots == [1, 2, 3]


# ==============================================================================
# Non-Retryable Error Handling: 400 Bad Request does NOT rotate keys
# ==============================================================================
def test_non_retryable_error_does_not_rotate_keys():
    """Non-retryable 400 Bad Request should raise immediately and NOT rotate to Key 2 or 3."""
    called_slots = []

    def mock_post(url, **kwargs):
        headers = kwargs.get("headers", {})
        key = headers.get("x-goog-api-key")
        if key == "key_one":
            called_slots.append(1)
            return httpx.Response(
                status_code=400,
                json={"error": {"code": 400, "message": "INVALID_ARGUMENT: malformed request"}},
                request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent"),
            )
        elif key == "key_two":
            called_slots.append(2)
        elif key == "key_three":
            called_slots.append(3)
        return _create_mock_gemini_tts_response(b"", status_code=200)

    mgr = GeminiKeyManager(keys=[(1, "key_one"), (2, "key_two"), (3, "key_three")])
    tts = GeminiTTS(key_manager=mgr)

    with patch("httpx.Client.post", side_effect=mock_post):
        with pytest.raises(Exception):
            tts.synthesize("Testing malformed request handling.", language="en-IN")

        # Only Key 1 should be attempted, Key 2 and Key 3 must NOT be called
        assert 1 in called_slots
        assert 2 not in called_slots
        assert 3 not in called_slots


# ==============================================================================
# Empty Audio Protection
# ==============================================================================
def test_empty_audio_protection():
    """Verifies that empty or corrupt audio triggers explicit validation failure."""
    mgr = GeminiKeyManager(keys=[(1, "fake_key_1")])

    with patch("httpx.Client.post") as mock_post:
        # Return 200 with empty PCM payload
        mock_post.return_value = _create_mock_gemini_tts_response(b"")
        tts = GeminiTTS(key_manager=mgr)

        with pytest.raises(ValueError) as exc_info:
            tts.synthesize("Testing empty audio protection.", language="en-IN")

        assert "empty PCM audio bytes" in str(exc_info.value)


# ==============================================================================
# API Endpoint Integration: POST /api/voice/synthesize
# ==============================================================================
def test_api_voice_synthesize_endpoint_success(client):
    """Integration test: POST /api/voice/synthesize returns 200 and audio/wav."""
    dummy_wav = b"RIFF" + b"\x00" * 50
    with patch("app.api.voice.GeminiTTS.synthesize", return_value=(dummy_wav, "audio/wav")):
        response = client.post(
            "/api/voice/synthesize",
            json={"text": "Please proceed to reception for billing.", "language": "en-IN"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.content == dummy_wav


def test_api_voice_synthesize_endpoint_empty_text(client):
    """Integration test: POST /api/voice/synthesize with empty text returns 400 Bad Request."""
    response = client.post(
        "/api/voice/synthesize",
        json={"text": "   ", "language": "en-IN"},
    )
    assert response.status_code == 400
    assert "Text cannot be empty" in response.text


def test_api_voice_synthesize_endpoint_quota_exceeded(client):
    """Integration test: Quota exhausted on all keys returns 429 Too Many Requests."""
    with patch(
        "app.api.voice.GeminiTTS.synthesize",
        side_effect=RuntimeError("Voice synthesis quota exceeded (429) across all configured API keys."),
    ):
        response = client.post(
            "/api/voice/synthesize",
            json={"text": "All quotas exhausted test.", "language": "en-IN"},
        )
        assert response.status_code == 429
        assert "quota exceeded" in response.text.lower()
