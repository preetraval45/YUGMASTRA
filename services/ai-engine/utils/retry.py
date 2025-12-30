import functools
import logging
from typing import Callable, Any, Optional, Type
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import asyncio

logger = logging.getLogger(__name__)


def retry_on_failure(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
    exceptions: tuple = (Exception,),
    log_level: int = logging.WARNING
):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, log_level),
        after=after_log(logger, log_level)
    )


def async_retry_on_failure(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
    exceptions: tuple = (Exception,),
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {str(e)}"
                        )

            raise last_exception

        return wrapper
    return decorator


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def _should_attempt_reset(self) -> bool:
        import time
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.timeout
        )
