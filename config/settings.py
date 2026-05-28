"""
config/settings.py
==================
Central configuration for the Energy Price Prediction project.

All runtime settings are read from environment variables with safe defaults,
making the project runnable locally (SQLite) and in production (PostgreSQL).
"""

import os
from pathlib import Path

# ── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Database ─────────────────────────────────────────────────────────────────
# Set DATABASE_URL in your .env file or shell to use PostgreSQL:
#   postgresql+psycopg2://energy_user:energy_pass@localhost:5432/energy_db
#
# If not set (or empty string from CI secrets), a local SQLite file is used automatically.
_raw_db_url = os.getenv("DATABASE_URL", "").strip()
DATABASE_URL: str = (
    _raw_db_url if _raw_db_url else f"sqlite:///{PROJECT_ROOT / 'data' / 'energy.db'}"
)

# ── API Keys ─────────────────────────────────────────────────────────────────
# ENTSO-E requires a free API token:  https://transparency.entsoe.eu/
ENTSOE_API_TOKEN: str = os.getenv("ENTSOE_API_TOKEN", "")

# OTE-CR is publicly scraped – no key needed.

# ── Data collection coverage ─────────────────────────────────────────────────
# How many calendar days of historical data to fetch on first run.
INITIAL_HISTORY_DAYS: int = int(os.getenv("INITIAL_HISTORY_DAYS", "2000"))

# ── Weather locations (Czech Republic) ───────────────────────────────────────
# Czech electricity prices apply nationwide, so we sample multiple locations
# and average the meteorological features to get a representative country signal.
WEATHER_LOCATIONS: list[dict] = [
    {"name": "Praha",   "latitude": 50.0755, "longitude": 14.4378},
    {"name": "Brno",    "latitude": 49.1951, "longitude": 16.6068},
    {"name": "Ostrava", "latitude": 49.8209, "longitude": 18.2625},
    {"name": "Plzeň",   "latitude": 49.7384, "longitude": 13.3736},
]

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgboost_price_model.joblib"
MODEL_VERSION: str = "1.0.0"

# ── Lag features ──────────────────────────────────────────────────────────────
# Hours to lag price for autoregressive features.
LAG_HOURS: list[int] = [24, 48, 168]   # 1 day, 2 days, 1 week

# ── Reports ──────────────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Timezone ─────────────────────────────────────────────────────────────────
TIMEZONE: str = "Europe/Prague"  # CZ is always in one timezone (CET/CEST)
