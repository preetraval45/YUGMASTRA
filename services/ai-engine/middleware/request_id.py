import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        # Get existing request ID or generate new one
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())

        # Store request ID in request state
        request.state.request_id = request_id

        # Add to logging context
        logger_extra = {"request_id": request_id}

        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers[self.header_name] = request_id

            return response

        except Exception as e:
            logger.error(
                f"Error processing request {request_id}: {str(e)}",
                extra=logger_extra
            )
            raise


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
