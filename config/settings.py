# config/settings.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    default_provider: str = "openrouter"
    default_model: str = "mistralai/devstral-2512:free"
    
    class Config:
        env_file = ".env"