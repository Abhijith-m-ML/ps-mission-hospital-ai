"""Multilingual language definitions, detection, and formatting utilities
for the Hospital AI Assistant (supporting English, Malayalam, and Hindi).
"""
import re
from typing import Optional

# Supported Language Identifiers
LANG_EN = "en-IN"
LANG_ML = "ml-IN"
LANG_HI = "hi-IN"
LANG_AUTO = "auto"

SUPPORTED_LANGUAGES = {LANG_EN, LANG_ML, LANG_HI}

LANGUAGE_LABELS = {
    LANG_EN: "English",
    LANG_ML: "മലയാളം",
    LANG_HI: "हिन्दी",
    LANG_AUTO: "Auto Detect",
}

# Unicode Ranges for Script Detection
# Malayalam: U+0D00 to U+0D7F
MALAYALAM_PATTERN = re.compile(r"[\u0D00-\u0D7F]")
# Devanagari (Hindi): U+0900 to U+097F
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def detect_language(text: Optional[str]) -> str:
    """Detects the primary language of the provided text by analyzing Unicode scripts.
    
    Args:
        text: Query or response string.
        
    Returns:
        str: 'ml-IN' for Malayalam, 'hi-IN' for Hindi, or 'en-IN' for English/default.
    """
    if not text:
        return LANG_EN

    # Count script characters
    ml_matches = len(MALAYALAM_PATTERN.findall(text))
    hi_matches = len(DEVANAGARI_PATTERN.findall(text))

    if ml_matches > 0 and ml_matches >= hi_matches:
        return LANG_ML
    elif hi_matches > 0:
        return LANG_HI

    return LANG_EN


def resolve_language(
    requested_language: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """Resolves a client language code ('en-IN', 'ml-IN', 'hi-IN', or 'auto')
    to a concrete supported language code, auto-detecting from text if requested.
    
    Args:
        requested_language: Client requested language code or name.
        text: Text to analyze when language is 'auto' or unspecified.
        
    Returns:
        str: One of 'en-IN', 'ml-IN', or 'hi-IN'.
    """
    code = (requested_language or "").strip().lower()

    if code in ("ml-in", "ml", "malayalam", "mal"):
        return LANG_ML
    if code in ("hi-in", "hi", "hindi", "hin"):
        return LANG_HI
    if code in ("en-in", "en", "english", "eng"):
        return LANG_EN

    # Auto detect mode or unspecified
    if code in ("auto", "autodetect", "detect", ""):
        return detect_language(text)

    return LANG_EN


def get_language_prompt_instruction(language_code: str) -> str:
    """Returns grounded prompt guidance for generating the hospital response
    in the designated language without translating proper names or schedules incorrectly.
    """
    if language_code == LANG_ML:
        return (
            "LANGUAGE INSTRUCTION: Answer the patient's inquiry in clear, natural, and polite Malayalam (മലയാളത്തിൽ വ്യക്തമായി മറുപടി നൽകുക).\n"
            "- CRITICAL RULE: DO NOT distort or invent doctor names, qualifications (MBBS, MD, MS, etc.), consultation hours, phone numbers, or URLs.\n"
            "- Keep doctor names prominently readable (e.g. 'ഡോ. ആനി ഷീല' or 'Dr. Sr. Annie Sheela')."
        )
    elif language_code == LANG_HI:
        return (
            "LANGUAGE INSTRUCTION: Answer the patient's inquiry in clear, natural, and polite Hindi (हिन्दी में स्पष्ट उत्तर दें).\n"
            "- CRITICAL RULE: DO NOT distort or invent doctor names, qualifications (MBBS, MD, MS, etc.), consultation hours, phone numbers, or URLs.\n"
            "- Keep doctor names prominently readable (e.g. 'डॉ. एनी शीला' or 'Dr. Sr. Annie Sheela')."
        )
    else:
        return (
            "LANGUAGE INSTRUCTION: Answer the patient's inquiry in clear, compassionate, and professional English."
        )
