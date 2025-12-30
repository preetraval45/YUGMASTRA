import asyncio
import json
from typing import AsyncGenerator, Any, Dict
from sse_starlette.sse import EventSourceResponse
import logging

logger = logging.getLogger(__name__)


async def stream_generator(
    text: str,
    chunk_size: int = 10,
    delay: float = 0.01
) -> AsyncGenerator[str, None]:
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield chunk
        await asyncio.sleep(delay)


async def stream_llm_response(
    llm_func: callable,
    *args,
    **kwargs
) -> AsyncGenerator[Dict[str, Any], None]:
    try:
        # Call LLM function
        response = await llm_func(*args, **kwargs)

        # Check if response has streaming support
        if hasattr(response, '__aiter__'):
            # Stream from async iterator
            async for chunk in response:
                yield {
                    "type": "chunk",
                    "data": chunk,
                    "done": False
                }
        else:
            # Simulate streaming for non-streaming responses
            text = response.get("text", "") if isinstance(response, dict) else str(response)
            async for chunk in stream_generator(text):
                yield {
                    "type": "chunk",
                    "data": chunk,
                    "done": False
                }

        # Send completion signal
        yield {
            "type": "done",
            "data": None,
            "done": True
        }

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield {
            "type": "error",
            "data": str(e),
            "done": True
        }


def create_sse_response(
    generator: AsyncGenerator
) -> EventSourceResponse:
    async def event_generator():
        try:
            async for item in generator:
                yield {
                    "event": item.get("type", "message"),
                    "data": json.dumps(item.get("data")),
                    "id": item.get("id", ""),
                }
        except Exception as e:
            logger.error(f"SSE error: {str(e)}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())
