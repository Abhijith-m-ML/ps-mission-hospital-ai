"""Speech-to-Text (STT) implementation using Google's hosted Gemini API.
Transcribes browser microphone audio in English, Malayalam, and Hindi.
"""
import base64
import time
from typing import Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logging_config import logger
from app.voice.language import (
    LANG_AUTO,
    LANG_EN,
    LANG_HI,
    LANG_ML,
    detect_language,
    resolve_language,
)


class GeminiSTT:
    """Gemini Speech-to-Text provider leveraging native audio multimodal understanding."""

    def __init__(self, api_key: Optional[str] = None):
        raw_key = api_key or settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
        self.api_key = raw_key.strip() if raw_key else None
        # Models with active audio support in Generative Language API
        self.candidate_models = [
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
        ]

    def _normalize_mime_type(self, raw_mime: Optional[str], audio_bytes: bytes) -> str:
        """Determines valid audio MIME type accepted by Gemini inline data."""
        if raw_mime:
            clean = raw_mime.split(";")[0].strip().lower()
            if clean in ("audio/webm", "audio/wav", "audio/x-wav", "audio/wave", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/aac", "audio/flac"):
                if clean in ("audio/x-wav", "audio/wave"):
                    return "audio/wav"
                if clean == "audio/mpeg":
                    return "audio/mp3"
                return clean

        # Magic bytes heuristic
        if audio_bytes.startswith(b"RIFF"):
            return "audio/wav"
        elif audio_bytes.startswith(b"\x1aE\xdf\xa3"):
            return "audio/webm"
        elif audio_bytes.startswith(b"OggS"):
            return "audio/ogg"
        elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
            return "audio/mp3"

        return "audio/webm"

    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: Optional[str] = None,
        language: Optional[str] = LANG_AUTO,
    ) -> Tuple[str, str]:
        """Transcribes recorded audio to text using Gemini.
        
        Args:
            audio_bytes: Raw audio binary data from browser microphone.
            mime_type: Audio MIME type (e.g. 'audio/webm', 'audio/wav').
            language: Requested language hint ('en-IN', 'ml-IN', 'hi-IN', or 'auto').
            
        Returns:
            Tuple[str, str]: (Transcribed text, detected/confirmed language code).
        """
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in backend/.env."
            )

        if not audio_bytes or len(audio_bytes) < 100:
            raise ValueError("Audio payload is empty or too short to transcribe.")

        resolved_mime = self._normalize_mime_type(mime_type, audio_bytes)
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Select prompt according to requested language hint
        lang_norm = (language or "").strip().lower()
        if lang_norm in ("ml-in", "ml", "malayalam"):
            prompt = (
                "You are an expert audio speech-to-text transcribing system for hospital and medical inquiries. "
                "The user is speaking Malayalam (മലയാളം), which may be pure Malayalam, colloquial Malayalam, or mixed Malayalam and English (Manglish) with medical symptoms or doctor names. "
                "Transcribe the spoken audio verbatim and accurately. Use Malayalam script (മലയാളം ലിപിയിൽ) for Malayalam words and English script for English words/acronyms (e.g. OP, doctor, Cardiology, leg pain). "
                "Do not summarize, do not translate, do not add notes. Return ONLY the transcribed text verbatim."
            )
        elif lang_norm in ("hi-in", "hi", "hindi"):
            prompt = (
                "You are an expert audio speech-to-text transcribing system for hospital inquiries. "
                "The user is speaking Hindi (हिन्दी), which may include English medical terms or doctor names. "
                "Transcribe the spoken audio verbatim and accurately into Devanagari script (देवनागरी लिपि में), using English script for English medical words if spoken. "
                "Do not summarize, do not translate, do not add notes. Return ONLY the transcribed text verbatim."
            )
        elif lang_norm in ("en-in", "en", "english"):
            prompt = (
                "You are an expert audio speech-to-text transcribing system for hospital inquiries. "
                "The user is speaking Indian English, which may include Indian doctor names, hospital departments, and medical symptoms. "
                "Transcribe the spoken English audio verbatim and accurately without changing any words or symptoms. "
                "Do not summarize, do not add notes. Return ONLY the transcribed text verbatim."
            )
        else:
            prompt = (
                "You are an expert audio speech-to-text transcribing system for hospital inquiries. "
                "The user may speak in Malayalam, English, Hindi, or a mix (such as Manglish or Hinglish). "
                "Accurately transcribe the spoken audio verbatim: transcribe Malayalam in Malayalam script, Hindi in Devanagari script, and English/medical terms in English script. "
                "Do not summarize, do not translate, do not add notes. Return ONLY the transcribed text verbatim."
            )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": resolved_mime,
                                "data": b64_audio,
                            }
                        },
                        {
                            "text": prompt,
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
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
                        raw_text = parts[0].get("text", "").strip()

                        # Strip markdown quotes or conversational prefixes if present
                        cleaned_text = raw_text.strip('"`\'\n ')
                        if cleaned_text.lower().startswith("transcript:"):
                            cleaned_text = cleaned_text[len("transcript:"):].strip()
                        elif cleaned_text.lower().startswith("transcription:"):
                            cleaned_text = cleaned_text[len("transcription:"):].strip()

                        logger.info(
                            "Gemini STT succeeded in %.3fs using %s (bytes=%d, mime=%s).",
                            duration,
                            model,
                            len(audio_bytes),
                            resolved_mime,
                        )

                        # Determine language of transcribed text
                        detected_lang = detect_language(cleaned_text)
                        final_lang = resolve_language(language, cleaned_text)
                        if final_lang == LANG_EN and detected_lang != LANG_EN:
                            final_lang = detected_lang

                        return cleaned_text, final_lang

                    elif resp.status_code in (429, 503):
                        logger.warning("Gemini STT model %s busy (HTTP %d). Trying next model...", model, resp.status_code)
                        last_error = RuntimeError(f"Gemini model {model} temporarily busy.")
                        continue
                    else:
                        logger.warning("Gemini STT model %s error %d: %s", model, resp.status_code, resp.text[:200])
                        last_error = RuntimeError(f"Gemini STT error (status {resp.status_code}).")
                        continue

            except httpx.TimeoutException:
                logger.warning("Gemini STT model %s timed out. Trying next model...", model)
                last_error = TimeoutError("Gemini STT request timed out.")
                continue
            except Exception as err:
                logger.warning("Gemini STT model %s failed: %s", model, err)
                last_error = err
                continue

        if last_error:
            raise last_error
        raise RuntimeError("Failed to transcribe audio with Gemini STT.")
