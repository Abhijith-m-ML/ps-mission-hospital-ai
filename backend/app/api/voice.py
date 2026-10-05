"""FastAPI API endpoints for multilingual voice input (STT) and output (TTS)."""
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status

from app.core.logging_config import logger
from app.models.schemas import VoiceSynthesizeRequest, VoiceTranscribeResponse
from app.voice.language import resolve_language
from app.voice.stt import GeminiSTT
from app.voice.tts import GeminiTTS

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post(
    "/transcribe",
    response_model=VoiceTranscribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe spoken audio from microphone using Gemini STT",
)
async def transcribe_voice(
    audio: UploadFile = File(..., description="Recorded audio binary from browser"),
    language: Optional[str] = Form(default="auto", description="Requested language or 'auto'"),
) -> VoiceTranscribeResponse:
    """Receives audio file from browser microphone, performs Speech-to-Text via
    Gemini multimodal understanding, and returns transcribed text alongside detected language.
    """
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio payload received.",
            )

        content_type = audio.content_type or "audio/webm"
        stt = GeminiSTT()
        text, detected_lang = stt.transcribe(
            audio_bytes=audio_bytes,
            mime_type=content_type,
            language=language,
        )

        return VoiceTranscribeResponse(
            text=text,
            transcript=text,
            language=detected_lang,
        )

    except HTTPException:
        raise
    except ValueError as err:
        logger.warning("Voice transcription validation error: %s", err)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except TimeoutError as err:
        logger.warning("Voice transcription timed out: %s", err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Speech recognition request timed out. Please try speaking again.",
        )
    except Exception as err:
        logger.exception("Voice transcription failed: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transcribe speech. Please try again.",
        )


@router.post(
    "/synthesize",
    status_code=status.HTTP_200_OK,
    summary="Synthesize answer text to speech audio using Gemini TTS",
)
def synthesize_voice(request: VoiceSynthesizeRequest) -> Response:
    """Converts hospital answer text into playable WAV audio using Gemini TTS
    in the requested language (English, Malayalam, or Hindi).
    """
    clean_text = (request.text or "").strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty.",
        )

    try:
        tts = GeminiTTS()
        audio_bytes, mime_type = tts.synthesize(
            text=clean_text,
            language=request.language or "en-IN",
        )

        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "Content-Disposition": 'inline; filename="hospital_answer.wav"',
                "Cache-Control": "no-cache",
            },
        )

    except HTTPException:
        raise
    except ValueError as err:
        logger.warning("Voice synthesis validation error: %s", err)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except TimeoutError as err:
        logger.warning("Voice synthesis timed out across all keys: %s", err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Speech synthesis request timed out. Please try again.",
        )
    except ConnectionError as err:
        logger.error("Voice synthesis connection error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The voice synthesis service is temporarily unreachable. Please check network connectivity.",
        )
    except Exception as err:
        logger.exception("Voice synthesis failed with %s: %s", type(err).__name__, err)
        err_msg = str(err)
        if any(term in err_msg.lower() for term in ("quota", "429", "rate limit", "resource_exhausted")):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Voice synthesis service quota exceeded across all configured API keys. Please try again shortly.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synthesize voice response: {err_msg}",
        )
