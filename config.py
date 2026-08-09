from decimal import Decimal
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    REPORTS_FOLDER = BASE_DIR / "reports"
    SAMPLE_DATA_FOLDER = BASE_DIR / "sample_data"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"
    OPENAI_TIMEOUT_SECONDS = float(
        os.getenv("OPENAI_TIMEOUT_SECONDS", "12")
    )
    OPENAI_MAX_OUTPUT_TOKENS = int(
        os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "500")
    )
    OPENAI_MODEL_PRICING = {
        "input_per_million": Decimal("0.20"),
        "cached_input_per_million": Decimal("0.02"),
        "output_per_million": Decimal("1.25"),
    }

    INSTANCE_FOLDER = BASE_DIR / "instance"
    GPU_USAGE_DATABASE = INSTANCE_FOLDER / "regressai.sqlite"
    RUNPOD_ENABLED = os.getenv("RUNPOD_ENABLED", "false").lower() == "true"
    RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
    RUNPOD_WAIT_MILLISECONDS = int(os.getenv("RUNPOD_WAIT_MILLISECONDS", "90000"))
    RUNPOD_HTTP_TIMEOUT_SECONDS = float(os.getenv("RUNPOD_HTTP_TIMEOUT_SECONDS", "100"))
    RUNPOD_EXECUTION_TIMEOUT_SECONDS = int(os.getenv("RUNPOD_EXECUTION_TIMEOUT_SECONDS", "60"))
    GPU_MAX_ENCODED_PAYLOAD_BYTES = int(os.getenv("GPU_MAX_ENCODED_PAYLOAD_BYTES", str(15 * 1024 * 1024)))
    GPU_MAX_DECOMPRESSED_BYTES = int(os.getenv("GPU_MAX_DECOMPRESSED_BYTES", str(256 * 1024 * 1024)))
    GPU_MAX_BOOTSTRAP_ITERATIONS = int(os.getenv("GPU_MAX_BOOTSTRAP_ITERATIONS", "10000"))
    GPU_MIN_BOOTSTRAP_ITERATIONS = int(os.getenv("GPU_MIN_BOOTSTRAP_ITERATIONS", "2000"))
    GPU_MIN_WORK_UNITS = int(os.getenv("GPU_MIN_WORK_UNITS", "10000000"))
    CPU_FALLBACK_MAX_WORK_UNITS = int(os.getenv("CPU_FALLBACK_MAX_WORK_UNITS", "25000000"))
    GPU_PRICE_PER_SECOND_USD = Decimal(os.getenv("GPU_PRICE_PER_SECOND_USD", "0.0002"))
    GPU_RESERVED_COST_USD = Decimal(os.getenv("GPU_RESERVED_COST_USD", "0.012"))
    GPU_DAILY_USER_LIMIT = int(os.getenv("GPU_DAILY_USER_LIMIT", "3"))
    GPU_MONTHLY_USER_LIMIT = int(os.getenv("GPU_MONTHLY_USER_LIMIT", "30"))
    GPU_GLOBAL_DAILY_BUDGET_USD = Decimal(os.getenv("GPU_GLOBAL_DAILY_BUDGET_USD", "2"))
    GPU_GLOBAL_MONTHLY_BUDGET_USD = Decimal(os.getenv("GPU_GLOBAL_MONTHLY_BUDGET_USD", "25"))
    GPU_GLOBAL_MAX_IN_FLIGHT = int(os.getenv("GPU_GLOBAL_MAX_IN_FLIGHT", "1"))
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
    BENCHMARK_CAMPAIGN_BUDGET_USD = Decimal(os.getenv("BENCHMARK_CAMPAIGN_BUDGET_USD", "1"))
