"""Text-to-Speech (TTS) implementation using Google's Gemini API.
Converts hospital assistant answers to playable WAV audio in English, Malayalam, and Hindi.
"""
import base64
import io
import re
import time
import wave
from typing import Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logging_config import logger
from app.voice.language import (
    LANG_EN,
    LANG_HI,
    LANG_ML,
    resolve_language,
)


class GeminiTTS:
    """Gemini Text-to-Speech provider synthesizing audio via Gemini speech generation."""

    def __init__(self, api_key: Optional[str] = None):
        raw_key = api_key or settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
        self.api_key = raw_key.strip() if raw_key else None
        # Models with active TTS support
        self.candidate_models = [
            "gemini-2.5-flash-preview-tts",
            "gemini-3.1-flash-tts-preview",
            "gemini-2.5-pro-preview-tts",
        ]

    def _clean_text_for_speech(self, text: str) -> str:
        """Removes markdown symbols, URLs, and source tags for natural speech flow."""
        if not text:
            return ""

        cleaned = text.strip()

        # Remove markdown code blocks
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
        # Convert links [Label](url) to just Label
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        # Remove raw URLs
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        # Remove [SOURCE ID: ...] citations
        cleaned = re.sub(r"\[SOURCE\s*ID:[^\]]+\]", "", cleaned, flags=re.IGNORECASE)
        # Remove bold and italic markers
        cleaned = re.sub(r"[*_~`#]", "", cleaned)
        # Replace list dashes with commas or spaces
        cleaned = re.sub(r"^\s*[-•*]\s+", "", cleaned, flags=re.MULTILINE)
        # Collapse excessive newlines or whitespace
        cleaned = re.sub(r"\n+", ". ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Budget length to max 600 characters for optimal TTS response time
        if len(cleaned) > 600:
            last_period = cleaned[:600].rfind(".")
            if last_period > 200:
                cleaned = cleaned[:last_period + 1]
            else:
                cleaned = cleaned[:600]

        return cleaned.strip()

    def _convert_pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
        """Wraps raw 16-bit linear PCM audio into a standard RIFF WAV container."""
        if pcm_bytes.startswith(b"RIFF"):
            return pcm_bytes

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit (2 bytes per sample)
            wf.setframerate(sample_rate)  # 24,000 Hz standard for Gemini TTS
            wf.writeframes(pcm_bytes)

        return wav_buffer.getvalue()

    def synthesize(
        self,
        text: str,
        language: Optional[str] = LANG_EN,
    ) -> Tuple[bytes, str]:
        """Synthesizes given text into playable WAV audio.
        
        Args:
            text: Text to read aloud.
            language: Language code ('en-IN', 'ml-IN', or 'hi-IN').
            
        Returns:
            Tuple[bytes, str]: (WAV audio binary data, MIME type 'audio/wav').
        """
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in backend/.env."
            )

        speech_text = self._clean_text_for_speech(text)
        if not speech_text:
            raise ValueError("Input text is empty after cleaning.")

        resolved_lang = resolve_language(language, speech_text)

        # Choose natural voice for the target language
        voice_name = "Puck"
        if resolved_lang == LANG_ML:
            voice_name = "Kore"
        elif resolved_lang == LANG_HI:
            voice_name = "Aoede"

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": speech_text,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name,
                        }
                    }
                },
            },
        }

        last_error = None
        for model in self.candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            start_time = time.time()
            try:
                with httpx.Client(timeout=25.0) as client:
                    resp = client.post(url, json=payload)
                    duration = time.time() - start_time

                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            continue
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if not parts:
                            continue

                        inline_data = parts[0].get("inlineData")
                        if not inline_data or "data" not in inline_data:
                            continue

                        raw_b64 = inline_data["data"]
                        raw_pcm = base64.b64decode(raw_b64)
                        wav_data = self._convert_pcm_to_wav(raw_pcm, sample_rate=24000)

                        logger.info(
                            "Gemini TTS succeeded in %.3fs using %s (chars=%d, bytes=%d, lang=%s).",
                            duration,
                            model,
                            len(speech_text),
                            len(wav_data),
                            resolved_lang,
                        )
                        return wav_data, "audio/wav"

                    elif resp.status_code in (429, 503):
                        logger.warning("Gemini TTS model %s busy (HTTP %d). Trying next model...", model, resp.status_code)
                        last_error = RuntimeError(f"Gemini TTS model {model} temporarily busy.")
                        continue
                    else:
                        logger.warning("Gemini TTS model %s error %d: %s", model, resp.status_code, resp.text[:200])
                        last_error = RuntimeError(f"Gemini TTS error (status {resp.status_code}).")
                        continue

            except httpx.TimeoutException:
                logger.warning("Gemini TTS model %s timed out. Trying next model...", model)
                last_error = TimeoutError("Gemini TTS request timed out.")
                continue
            except Exception as err:
                logger.warning("Gemini TTS model %s failed: %s", model, err)
                last_error = err
                continue

        if last_error:
            raise last_error
        raise RuntimeError("Failed to synthesize audio with Gemini TTS.")
