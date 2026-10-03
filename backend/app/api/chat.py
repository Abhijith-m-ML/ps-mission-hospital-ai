from fastapi import APIRouter, HTTPException, status
from app.core.logging_config import logger
from app.models.schemas import ChatRequest, ChatResponse
from app.rag.rag_service import RAGService

router = APIRouter(prefix="", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a hospital inquiry and receive a grounded factual answer with verified sources",
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """End-to-end RAG conversational endpoint:
    Receives a patient/visitor message, executes Pinecone semantic retrieval within
    P.S. Mission Hospital records, formats grounded factual context, queries the hosted LLM,
    and returns an accurate, verified answer alongside official source citations.
    """
    clean_message = (request.message or "").strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    try:
        service = RAGService()
        response = service.answer_question(
            message=clean_message,
            history=request.history,
            session_id=request.session_id,
            language=request.language,
        )
        return response

    except ValueError as err:
        logger.warning("Validation or configuration error in chat: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except TimeoutError as err:
        logger.warning("Timeout error in chat: %s", err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The request to the AI service timed out. Please try again.",
        )
    except ConnectionError as err:
        logger.error("Connection error in chat: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI service is temporarily unavailable. Please verify network connectivity.",
        )
    except Exception as err:
        logger.exception("Unexpected error processing chat message: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the hospital response. Please try again later.",
        )
