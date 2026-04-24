import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_DATABASE: str = os.getenv("DB_DATABASE", "restaurant_orders")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_PORT: str = os.getenv("DB_PORT", "5432")

    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")

    GMAIL_USER: str = os.getenv("GMAIL_USER", "")
    GMAIL_PASSWORD: str = os.getenv("GMAIL_PASSWORD", "")

    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))


settings = Settings()
