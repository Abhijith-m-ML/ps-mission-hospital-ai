"""Multilingual voice support package for Hospital AI Assistant."""
from app.voice.language import (
    LANG_AUTO,
    LANG_EN,
    LANG_HI,
    LANG_ML,
    SUPPORTED_LANGUAGES,
    detect_language,
    get_language_prompt_instruction,
    resolve_language,
)
from app.voice.stt import GeminiSTT
from app.voice.tts import GeminiTTS

__all__ = [
    "LANG_AUTO",
    "LANG_EN",
    "LANG_HI",
    "LANG_ML",
    "SUPPORTED_LANGUAGES",
    "detect_language",
    "resolve_language",
    "get_language_prompt_instruction",
    "GeminiSTT",
    "GeminiTTS",
]
