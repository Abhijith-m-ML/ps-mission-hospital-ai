from datetime import datetime, timezone
from app.core.config import settings
from app.models.schemas import HealthResponse


class HealthService:
    """Service providing operational health check status."""

    @staticmethod
    def get_system_health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            app_name=settings.PROJECT_NAME,
            environment=settings.ENVIRONMENT,
            version="0.1.0",
            timestamp=datetime.now(timezone.utc),
        )
