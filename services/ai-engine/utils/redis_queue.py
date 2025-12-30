import json
import uuid
import time
import asyncio
from typing import Dict, Any, Optional, Callable
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)


class RedisQueue:
    def __init__(
        self,
        redis_url: str,
        queue_name: str = "ai_requests",
        processing_queue: str = "ai_processing",
        max_retries: int = 3
    ):
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.processing_queue = processing_queue
        self.max_retries = max_retries
        self.redis_client: Optional[aioredis.Redis] = None

    async def connect(self):
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Connected to Redis queue")

    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("Disconnected from Redis queue")

    async def enqueue(
        self,
        task_data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        await self.connect()

        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "data": task_data,
            "priority": priority,
            "created_at": time.time(),
            "retries": 0,
            "status": "queued"
        }
        await self.redis_client.zadd(
            self.queue_name,
            {json.dumps(task): -priority}
        )

        logger.info(f"Task {task_id} added to queue with priority {priority}")
        return task_id

    async def dequeue(self, timeout: int = 0) -> Optional[Dict[str, Any]]:
        await self.connect()
        tasks = await self.redis_client.zpopmin(self.queue_name, 1)

        if not tasks:
            if timeout > 0:
                await asyncio.sleep(0.1)
                return await self.dequeue(timeout - 0.1)
            return None

        task_json, _ = tasks[0]
        task = json.loads(task_json)
        task["status"] = "processing"
        task["started_at"] = time.time()

        await self.redis_client.hset(
            self.processing_queue,
            task["id"],
            json.dumps(task)
        )

        logger.info(f"Task {task['id']} dequeued for processing")
        return task

    async def complete_task(self, task_id: str, result: Any = None):
        await self.connect()

        # Get task from processing queue
        task_json = await self.redis_client.hget(self.processing_queue, task_id)
        if not task_json:
            logger.warning(f"Task {task_id} not found in processing queue")
            return

        task = json.loads(task_json)
        task["status"] = "completed"
        task["completed_at"] = time.time()
        task["result"] = result
        await self.redis_client.setex(
            f"completed:{task_id}",
            3600,
            json.dumps(task)
        )
        await self.redis_client.hdel(self.processing_queue, task_id)

        logger.info(f"Task {task_id} completed")

    async def fail_task(
        self,
        task_id: str,
        error: str,
        retry: bool = True
    ):
        await self.connect()

        # Get task from processing queue
        task_json = await self.redis_client.hget(self.processing_queue, task_id)
        if not task_json:
            logger.warning(f"Task {task_id} not found in processing queue")
            return

        task = json.loads(task_json)
        task["retries"] += 1
        task["last_error"] = error
        if retry and task["retries"] < self.max_retries:
            task["status"] = "queued"
            await self.redis_client.zadd(
                self.queue_name,
                {json.dumps(task): -(task["priority"] - task["retries"])}
            )
            logger.info(
                f"Task {task_id} failed, retry {task['retries']}/{self.max_retries}"
            )
        else:
            task["status"] = "failed"
            task["failed_at"] = time.time()
            await self.redis_client.setex(
                f"failed:{task_id}",
                86400,
                json.dumps(task)
            )
            logger.error(f"Task {task_id} permanently failed: {error}")
        await self.redis_client.hdel(self.processing_queue, task_id)

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        await self.connect()
        task_json = await self.redis_client.hget(self.processing_queue, task_id)
        if task_json:
            return json.loads(task_json)
        task_json = await self.redis_client.get(f"completed:{task_id}")
        if task_json:
            return json.loads(task_json)
        task_json = await self.redis_client.get(f"failed:{task_id}")
        if task_json:
            return json.loads(task_json)

        return None

    async def get_queue_length(self) -> int:
        await self.connect()
        return await self.redis_client.zcard(self.queue_name)

    async def get_processing_count(self) -> int:
        await self.connect()
        return await self.redis_client.hlen(self.processing_queue)


class QueueWorker:
    def __init__(
        self,
        queue: RedisQueue,
        handler: Callable,
        worker_id: Optional[str] = None
    ):
        self.queue = queue
        self.handler = handler
        self.worker_id = worker_id or str(uuid.uuid4())
        self.running = False

    async def start(self):
        self.running = True
        logger.info(f"Worker {self.worker_id} started")

        while self.running:
            try:
                task = await self.queue.dequeue(timeout=1)

                if task:
                    try:
                        result = await self.handler(task["data"])
                        await self.queue.complete_task(task["id"], result)

                    except Exception as e:
                        logger.error(
                            f"Worker {self.worker_id} error processing task {task['id']}: {str(e)}"
                        )
                        await self.queue.fail_task(task["id"], str(e))

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {str(e)}")
                await asyncio.sleep(1)

    async def stop(self):
        self.running = False
        logger.info(f"Worker {self.worker_id} stopped")
