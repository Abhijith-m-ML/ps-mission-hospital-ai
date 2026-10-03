from fastapi import APIRouter, status
from app.core.logging_config import logger
from app.models.schemas import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns the current operational status, environment, and timestamp of the backend API.",
)
def get_health() -> HealthResponse:
    logger.info("Health check endpoint accessed")
    return HealthService.get_system_health()
