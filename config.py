from decimal import Decimal
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
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
