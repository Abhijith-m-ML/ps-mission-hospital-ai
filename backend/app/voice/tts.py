"""Text-to-Speech (TTS) implementation using Google's Gemini API with 3-key fallback.
Converts hospital assistant answers to playable WAV audio in English, Malayalam, and Hindi.
"""
import base64
import io
import re
import time
import wave
from typing import Any, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logging_config import logger
from app.services.gemini_key_manager import GeminiKeyManager, get_gemini_key_manager
from app.voice.language import (
    LANG_EN,
    LANG_HI,
    LANG_ML,
    resolve_language,
)


class GeminiTTS:
    """Gemini Text-to-Speech provider synthesizing audio with multi-key fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        key_manager: Optional[GeminiKeyManager] = None,
        client: Optional[Any] = None,
    ):
        if api_key:
            self.key_manager = GeminiKeyManager(keys=[(1, api_key)])
        else:
            self.key_manager = key_manager or get_gemini_key_manager()

        # Models with active Gemini TTS support
        self.candidate_models = [
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        ]
        self._client = client

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

    def _verify_audio_payload(self, raw_pcm: bytes, wav_data: bytes, model: str) -> None:
        """Enforces empty audio protection by validating bytes, length, and RIFF container."""
        if not raw_pcm or len(raw_pcm) == 0:
            logger.error("TTS audio verification failure: raw PCM audio payload is empty (model=%s).", model)
            raise ValueError("Gemini TTS returned empty PCM audio bytes.")

        if not wav_data or len(wav_data) < 44:
            logger.error("TTS audio verification failure: WAV data too short (<44 bytes) (model=%s).", model)
            raise ValueError("Gemini TTS produced an incomplete WAV audio container.")

        if not wav_data.startswith(b"RIFF"):
            logger.error("TTS audio verification failure: WAV header missing RIFF magic bytes (model=%s).", model)
            raise ValueError("Gemini TTS audio container is corrupt (missing RIFF header).")

    def synthesize(
        self,
        text: str,
        language: Optional[str] = LANG_EN,
    ) -> Tuple[bytes, str]:
        """Synthesizes given text into playable WAV audio using resilient Gemini key fallback.
        
        Args:
            text: Text to read aloud.
            language: Language code ('en-IN', 'ml-IN', or 'hi-IN').
            
        Returns:
            Tuple[bytes, str]: (WAV audio binary data, MIME type 'audio/wav').
        """
        speech_text = self._clean_text_for_speech(text)
        if not speech_text:
            raise ValueError("Input text is empty after cleaning.")

        resolved_lang = resolve_language(language, speech_text)

        # Choose natural prebuilt voice for target language
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

        def _perform_tts(api_key: str, key_slot: int) -> Tuple[bytes, str]:
            # Log selected key slot index ONLY (NEVER the actual API key)
            logger.info("Gemini key slot: %d", key_slot)

            last_attempt_err = None
            for model in self.candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                headers = {"x-goog-api-key": api_key}
                start_time = time.time()
                try:
                    client_ctx = httpx.Client(timeout=25.0) if self._client is None else self._client
                    with client_ctx as client:
                        resp = client.post(url, headers=headers, json=payload)
                        duration = time.time() - start_time

                        if resp.status_code == 200:
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if not candidates:
                                logger.warning("Gemini TTS response missing candidates list (model=%s, slot=%d).", model, key_slot)
                                continue

                            parts = candidates[0].get("content", {}).get("parts", [])
                            if not parts:
                                logger.warning("Gemini TTS response missing content parts (model=%s, slot=%d).", model, key_slot)
                                continue

                            inline_data = parts[0].get("inlineData")
                            if not inline_data or "data" not in inline_data:
                                logger.warning("Gemini TTS response missing inlineData (model=%s, slot=%d).", model, key_slot)
                                continue

                            raw_b64 = inline_data["data"]
                            raw_pcm = base64.b64decode(raw_b64)
                            wav_data = self._convert_pcm_to_wav(raw_pcm, sample_rate=24000)

                            # Enforce empty audio protection
                            self._verify_audio_payload(raw_pcm, wav_data, model)

                            logger.info(
                                "Gemini TTS succeeded in %.3fs: model=%s key_slot=%d chars=%d bytes=%d lang=%s.",
                                duration,
                                model,
                                key_slot,
                                len(speech_text),
                                len(wav_data),
                                resolved_lang,
                            )
                            return wav_data, "audio/wav"

                        # Handle HTTP error statuses
                        logger.warning(
                            "Gemini TTS model %s returned status %d on key slot %d: %s",
                            model,
                            resp.status_code,
                            key_slot,
                            resp.text[:200],
                        )

                        # Trigger key manager fallback on retryable status codes
                        if resp.status_code in (401, 403, 429, 500, 502, 503, 504):
                            provider_msg = None
                            try:
                                data = resp.json()
                                if isinstance(data, dict):
                                    err_dict = data.get("error", {})
                                    if isinstance(err_dict, dict):
                                        provider_msg = err_dict.get("message")
                                    elif isinstance(err_dict, str):
                                        provider_msg = err_dict
                            except Exception:
                                pass
                            if not provider_msg and hasattr(resp, "text"):
                                provider_msg = resp.text[:400]
                            err = httpx.HTTPStatusError(f"HTTP {resp.status_code}: {provider_msg}", request=resp.request, response=resp)
                            err.provider_message = provider_msg
                            raise err

                        # Non-retryable 4xx on this model: try next candidate model
                        last_attempt_err = RuntimeError(f"Gemini TTS error (status {resp.status_code}): {resp.text[:150]}")
                        continue

                except (httpx.TimeoutException, TimeoutError) as err:
                    logger.warning("Gemini TTS model %s timed out on key slot %d.", model, key_slot)
                    last_attempt_err = err
                    raise  # Let key manager catch and retry on next slot

                except httpx.HTTPStatusError as err:
                    # Retryable HTTP status error: raise to key manager for slot fallback
                    raise

                except Exception as err:
                    logger.warning("Gemini TTS model %s failed on key slot %d: %s", model, key_slot, err)
                    last_attempt_err = err
                    continue

            if last_attempt_err:
                raise last_attempt_err
            raise RuntimeError(f"Gemini TTS failed across all candidate models on key slot {key_slot}.")

        return self.key_manager.execute_with_fallback(
            func=_perform_tts,
            operation_name="Gemini TTS",
            model=self.candidate_models[0],
            language=resolved_lang,
            text_length=len(speech_text),
        )
