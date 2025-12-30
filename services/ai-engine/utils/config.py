import os
import logging
from typing import Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    url: str = Field(..., env="DATABASE_URL")
    max_connections: int = Field(default=200, env="POSTGRES_MAX_CONNECTIONS")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)


class RedisConfig(BaseModel):
    url: str = Field(..., env="REDIS_URL")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    max_connections: int = Field(default=50)
    decode_responses: bool = True


class OllamaConfig(BaseModel):
    url: str = Field(default="http://ollama:11434", env="OLLAMA_URL")
    default_model: str = Field(default="llama2", env="OLLAMA_DEFAULT_MODEL")
    timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")
    keep_alive: str = Field(default="5m", env="OLLAMA_KEEP_ALIVE")


class AIEngineConfig(BaseModel):
    model_dir: str = Field(default="/app/models", env="MODEL_DIR")
    data_dir: str = Field(default="/app/data", env="DATA_DIR")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    workers: int = Field(default=4, env="WORKERS")
    max_requests: int = Field(default=1000, env="MAX_REQUESTS")
    timeout: int = Field(default=300, env="TIMEOUT")
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: int = Field(default=1, env="RETRY_DELAY")


class RateLimitConfig(BaseModel):
    enabled: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    window_ms: int = Field(default=60000, env="RATE_LIMIT_WINDOW_MS")
    max_requests: int = Field(default=100, env="RATE_LIMIT_MAX_REQUESTS")


class MonitoringConfig(BaseModel):
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="/app/logs/app.log", env="LOG_FILE")
    request_id_header: str = Field(default="X-Request-ID", env="REQUEST_ID_HEADER")


class SecurityConfig(BaseModel):
    enable_csrf: bool = Field(default=True, env="ENABLE_CSRF_PROTECTION")
    enable_helmet: bool = Field(default=True, env="ENABLE_HELMET")
    cors_origins: list = Field(default=["*"], env="CORS_ORIGIN")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")

    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v


class Settings(BaseSettings):
    database: DatabaseConfig = DatabaseConfig(
        url=os.getenv("DATABASE_URL", "postgresql://yugmastra:yugmastra_dev_password@postgres:5432/yugmastra")
    )
    redis: RedisConfig = RedisConfig(
        url=os.getenv("REDIS_URL", "redis://redis:6379/0")
    )
    ollama: OllamaConfig = OllamaConfig()
    ai_engine: AIEngineConfig = AIEngineConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    security: SecurityConfig = SecurityConfig()
    environment: str = Field(default="development", env="NODE_ENV")
    debug: bool = Field(default=False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def validate_environment() -> Settings:
    try:
        settings = Settings()

        os.makedirs(settings.ai_engine.model_dir, exist_ok=True)
        os.makedirs(settings.ai_engine.data_dir, exist_ok=True)
        log_dir = os.path.dirname(settings.monitoring.log_file)
        os.makedirs(log_dir, exist_ok=True)

        logger.info("Environment validation successful")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"Ollama URL: {settings.ollama.url}")
        logger.info(f"Redis URL: {settings.redis.url}")

        return settings

    except Exception as e:
        logger.error(f"Environment validation failed: {str(e)}")
        raise ValueError(f"Invalid environment configuration: {str(e)}")


settings: Optional[Settings] = None


def get_settings() -> Settings:
    global settings
    if settings is None:
        settings = validate_environment()
    return settings
