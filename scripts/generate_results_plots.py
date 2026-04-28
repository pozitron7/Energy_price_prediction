"""
scripts/generate_results_plots.py
==================================
Generate all static result plots for the Energy Price Prediction project.

This script loads actuals and predictions from the database, trains-/loads
the XGBoost model from disk, and produces four publication-ready PNG figures
saved under ``reports/figures/``.

Four figure types are produced:

1. **actual_vs_predicted.png** – Time-series comparison with weekend shading.
2. **forecast_next_day.png**   – Bar chart of next-day hourly price forecasts
   colour-coded by percentile rank.
3. **feature_importance.png**  – Horizontal bar chart of the top-N XGBoost
   feature importances.
4. **error_distribution.png**  – Histogram of prediction errors.

Run this script directly::

    python scripts/generate_results_plots.py

Or import and call :func:`generate_all_plots` from another module.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

matplotlib.use("Agg")  # headless rendering – no display required

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

_SQL_ACTUALS_AND_PREDS = """
SELECT
    f.timestamp_utc,
    f.price_eur_mwh              AS actual,
    p.predicted_price_eur        AS predicted
FROM fact_hourly_market_features f
JOIN model_predictions p
    ON f.timestamp_utc = p.target_timestamp_utc
WHERE f.price_eur_mwh IS NOT NULL
  AND p.model_version = :model_version
ORDER BY f.timestamp_utc
"""

_SQL_NEXT_DAY_PREDS = """
SELECT
    target_timestamp_utc AS timestamp_utc,
    predicted_price_eur
FROM model_predictions
WHERE model_version = :model_version
  AND target_timestamp_utc >  :start_utc
  AND target_timestamp_utc <  :end_utc
ORDER BY timestamp_utc
"""

_SQL_FEATURES = """
SELECT *
FROM fact_hourly_market_features
ORDER BY timestamp_utc DESC
LIMIT 500
"""


# ---------------------------------------------------------------------------
# Plot 1 – Actual vs Predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    actuals: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    output_path: Path,
    mae: float,
    rmse: float,
    timestamps: pd.DatetimeIndex | None = None,
) -> None:
    """Generate a time-series line plot comparing actual and predicted prices.

    Plots actual prices as a solid blue line and predicted prices as an orange
    dashed line.  If *timestamps* are provided, weekend hours are shaded with
    a translucent red band as a visual reference.  The plot title includes the
    MAE and RMSE metric values.

    Args:
        actuals: 1-D array of observed EUR/MWh prices.
        predictions: 1-D array of predicted EUR/MWh prices.
        output_path: Destination file path for the saved PNG.
        mae: Mean Absolute Error to display in the title.
        rmse: Root Mean Squared Error to display in the title.
        timestamps: Optional ``DatetimeIndex`` aligned with *actuals* /
            *predictions*.  Required for weekend shading and x-axis date
            labels.  If ``None``, integer indices are used on the x-axis.

    Returns:
        ``None`` – the figure is saved to *output_path* and closed.

    Example::

        plot_actual_vs_predicted(
            actuals=actual_arr,
            predictions=pred_arr,
            output_path=figures_dir / "actual_vs_predicted.png",
            mae=3.14,
            rmse=4.72,
            timestamps=ts_index,
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5))

    x = timestamps if timestamps is not None else np.arange(len(actuals))

    ax.plot(x, actuals, color="steelblue", linewidth=1.2, label="Actual (EUR/MWh)")
    ax.plot(
        x,
        predictions,
        color="darkorange",
        linewidth=1.0,
        linestyle="--",
        label="Predicted (EUR/MWh)",
    )

    # Weekend shading requires datetime timestamps
    if timestamps is not None:
        ts = pd.DatetimeIndex(timestamps)
        # Mark spans where weekday is Saturday (5) or Sunday (6)
        in_weekend = ts.dayofweek >= 5
        if in_weekend.any():
            # Group consecutive weekend hours into spans
            changes = np.diff(in_weekend.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            for s, e in zip(starts, ends):
                ax.axvspan(ts[s], ts[min(e, len(ts) - 1)], alpha=0.08, color="red", label="_weekend")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30)

    ax.set_title(
        f"Actual vs Predicted Electricity Prices | MAE = {mae:.2f} EUR/MWh | RMSE = {rmse:.2f} EUR/MWh",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Plot 2 – Next-Day Forecast Bar Chart
# ---------------------------------------------------------------------------

def plot_forecast_next_day(
    predictions_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a bar chart of predicted hourly prices for the next day.

    Bars are colour-coded by percentile rank of the predicted price:

    - **Red**  – price is above the 80th percentile (expensive hours).
    - **Green** – price is below the 20th percentile (cheap hours).
    - **Gray** – price is between the 20th and 80th percentile.

    Args:
        predictions_df: DataFrame with columns ``["timestamp_utc",
            "predicted_price_eur"]`` containing exactly 24 hourly rows for
            the next calendar day.  Fewer rows are handled gracefully.
        output_path: Destination file path for the saved PNG.

    Returns:
        ``None`` – the figure is saved to *output_path* and closed.

    Example::

        plot_forecast_next_day(
            predictions_df=next_day_df,
            output_path=figures_dir / "forecast_next_day.png",
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = predictions_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    prices = df["predicted_price_eur"].to_numpy()
    p20 = float(np.percentile(prices, 20))
    p80 = float(np.percentile(prices, 80))

    colours = []
    for p in prices:
        if p > p80:
            colours.append("firebrick")
        elif p < p20:
            colours.append("seagreen")
        else:
            colours.append("slategray")

    hours = df["timestamp_utc"].dt.hour.tolist()
    x_labels = [f"{h:02d}:00" for h in hours]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(prices)), prices, color=colours, edgecolor="white", linewidth=0.5)

    # Annotate each bar with its price value
    for rect, val in zip(bars, prices):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
        )

    ax.set_xticks(range(len(prices)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Predicted Price (EUR/MWh)")

    target_date = df["timestamp_utc"].dt.date.iloc[0] if not df.empty else "N/A"
    ax.set_title(
        f"Next-Day Hourly Price Forecast | {target_date}\n"
        f"green: < P20 ({p20:.1f})   red: > P80 ({p80:.1f})   (EUR/MWh)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Plot 3 – Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: object,
    feature_names: Sequence[str],
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Generate a horizontal bar chart of XGBoost feature importances.

    Extracts importances from ``model.feature_importances_`` (scikit-learn
    API) or falls back to ``model.get_score(importance_type='gain')`` for
    native XGBoost ``Booster`` objects.

    Args:
        model: A fitted XGBoost model object exposing either
            ``feature_importances_`` (``XGBRegressor`` / ``XGBClassifier``)
            or ``get_score()`` (native ``xgboost.Booster``).
        feature_names: Sequence of feature name strings aligned with the
            model's training columns.
        output_path: Destination file path for the saved PNG.
        top_n: Maximum number of features to display.  Defaults to ``15``.

    Returns:
        ``None`` – the figure is saved to *output_path* and closed.

    Raises:
        AttributeError: If the model exposes neither ``feature_importances_``
            nor ``get_score()``.

    Example::

        plot_feature_importance(
            model=xgb_model,
            feature_names=FEATURE_COLS,
            output_path=figures_dir / "feature_importance.png",
            top_n=15,
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Extract importance scores ---
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        names = list(feature_names)
    elif hasattr(model, "get_score"):
        score_dict: dict[str, float] = model.get_score(importance_type="gain")
        names = list(score_dict.keys())
        importances = np.array([score_dict[k] for k in names], dtype=float)
    else:
        raise AttributeError(
            "Model exposes neither `feature_importances_` nor `get_score()`. "
            "Please pass a fitted XGBoost model."
        )

    # Sort and trim to top_n
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in sorted_idx]
    top_vals = importances[sorted_idx]

    # Reverse so largest bar is at the top
    top_names = top_names[::-1]
    top_vals = top_vals[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.45)))
    bars = ax.barh(range(len(top_names)), top_vals, color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Importance (Gain)", fontsize=10)
    ax.set_title(
        f"XGBoost Feature Importances (Top {top_n})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Annotate bars with numeric values
    for bar, val in zip(bars, top_vals):
        ax.text(
            val + max(top_vals) * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Plot 4 – Error Distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(
    actuals: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    output_path: Path,
) -> None:
    """Generate a histogram of prediction errors (actual − predicted).

    A vertical dashed line is drawn at zero error and the mean error is
    annotated on the chart.  The distribution shape indicates systematic
    bias (non-zero mean) or heavy-tailed errors (skewness).

    Args:
        actuals: 1-D array of observed EUR/MWh prices.
        predictions: 1-D array of predicted EUR/MWh prices.
        output_path: Destination file path for the saved PNG.

    Returns:
        ``None`` – the figure is saved to *output_path* and closed.

    Example::

        plot_error_distribution(
            actuals=actual_arr,
            predictions=pred_arr,
            output_path=figures_dir / "error_distribution.png",
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors = np.asarray(actuals, dtype=float) - np.asarray(predictions, dtype=float)
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.85)

    # Zero-error reference line
    ax.axvline(0, color="black", linewidth=1.5, linestyle="-", label="Zero error")
    # Mean error annotation line
    ax.axvline(
        mean_err,
        color="firebrick",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean error = {mean_err:+.2f} EUR/MWh",
    )

    # Annotate mean error value on the plot
    ax.annotate(
        f"μ = {mean_err:+.2f}\nσ = {std_err:.2f}",
        xy=(mean_err, ax.get_ylim()[1] * 0.85),
        xytext=(mean_err + std_err * 0.5, ax.get_ylim()[1] * 0.85),
        fontsize=9,
        color="firebrick",
        arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.0),
    )

    ax.set_xlabel("Prediction Error (EUR/MWh)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(
        "Prediction Error Distribution (Actual − Predicted)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def plot_shap_summary(model, features: pd.DataFrame, output_path: Path) -> None:
    """Save a SHAP beeswarm summary plot explaining the price model.

    Shows how each feature (e.g. solar irradiance, wind, hour of day) pushes
    the predicted price up or down.

    Args:
        model: Fitted tree model (XGBoost).
        features: Sample of the feature matrix used for the explanation.
        output_path: Destination PNG path.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, features, show=False, max_display=15)
    plt.title("SHAP: impact of features on predicted price (EUR/MWh)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path)


def generate_all_plots(
    engine: Engine,
    model_path: Path,
    figures_dir: Path,
) -> list[str]:
    """Orchestrate loading data and generating all four result plots.

    Performs the following steps:

    1. Load actuals + predictions from DB and call
       :func:`plot_actual_vs_predicted`.
    2. Load next-day predictions (tomorrow's 24 hours) and call
       :func:`plot_forecast_next_day`.
    3. Load the trained XGBoost model from *model_path* and call
       :func:`plot_feature_importance`.
    4. Re-use the residuals from step 1 to call
       :func:`plot_error_distribution`.

    Args:
        engine: Connected SQLAlchemy engine.
        model_path: Path to the serialised XGBoost model (joblib file).
        figures_dir: Directory where PNG files will be saved.

    Returns:
        A list of absolute path strings for every PNG file that was
        successfully saved.

    Raises:
        FileNotFoundError: If *model_path* does not exist on disk.
        ValueError: If the database contains no overlapping actuals and
            predictions.
    """
    import joblib  # optional dependency – keep import local

    from config import settings

    model_path = Path(model_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    saved_paths: list[str] = []

    # ------------------------------------------------------------------
    # Step 1 – Load actuals + predictions
    # ------------------------------------------------------------------
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_SQL_ACTUALS_AND_PREDS),
            conn,
            params={"model_version": settings.MODEL_VERSION},
        )

    if df.empty:
        raise ValueError("No overlapping actuals/predictions found in database.")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    actual_arr = df["actual"].to_numpy()
    pred_arr = df["predicted"].to_numpy()
    ts_index = pd.DatetimeIndex(df["timestamp_utc"])

    from src.evaluation.metrics import mae as _mae, rmse as _rmse

    mae_val = _mae(actual_arr, pred_arr)
    rmse_val = _rmse(actual_arr, pred_arr)

    # ------------------------------------------------------------------
    # Plot 1
    # ------------------------------------------------------------------
    out1 = figures_dir / "actual_vs_predicted.png"
    plot_actual_vs_predicted(
        actuals=actual_arr,
        predictions=pred_arr,
        output_path=out1,
        mae=mae_val,
        rmse=rmse_val,
        timestamps=ts_index,
    )
    saved_paths.append(str(out1.resolve()))

    # ------------------------------------------------------------------
    # Plot 4 – Error distribution (built from same data while we have it)
    # ------------------------------------------------------------------
    out4 = figures_dir / "error_distribution.png"
    plot_error_distribution(
        actuals=actual_arr,
        predictions=pred_arr,
        output_path=out4,
    )
    saved_paths.append(str(out4.resolve()))

    # ------------------------------------------------------------------
    # Step 2 – Next-day forecast
    # ------------------------------------------------------------------
    # Next-day forecast = predictions after the latest published actual price.
    with engine.connect() as conn:
        last_actual = conn.execute(
            text("SELECT MAX(timestamp_utc) FROM raw_energy_prices")
        ).scalar()
    start_utc = str(last_actual) if last_actual else f"{date.today().isoformat()}T00:00:00"
    end_utc = "9999-12-31T00:00:00"

    with engine.connect() as conn:
        next_day_df = pd.read_sql(
            text(_SQL_NEXT_DAY_PREDS),
            conn,
            params={
                "model_version": settings.MODEL_VERSION,
                "start_utc": start_utc,
                "end_utc": end_utc,
            },
        )

    out2 = figures_dir / "forecast_next_day.png"
    if next_day_df.empty:
        logger.warning("No next-day predictions found; skipping forecast plot.")
    else:
        plot_forecast_next_day(predictions_df=next_day_df, output_path=out2)
        saved_paths.append(str(out2.resolve()))

    # ------------------------------------------------------------------
    # Step 3 – Feature importance
    # ------------------------------------------------------------------
    logger.info("Loading model from %s …", model_path)
    artefact = joblib.load(model_path)
    # ModelTrainer.save_model stores a dict artefact {"model", "feature_cols", ...}
    if isinstance(artefact, dict):
        model = artefact["model"]
        feature_names = list(artefact.get("feature_cols", []))
    else:
        model, feature_names = artefact, []

    if not feature_names:
        # Fallback: derive from the fact table column order
        with engine.connect() as conn:
            feat_df = pd.read_sql(text(_SQL_FEATURES), conn)
        non_feature_cols = {"timestamp_utc", "price_eur_mwh", "updated_at"}
        feature_names = [c for c in feat_df.columns if c not in non_feature_cols]

    out3 = figures_dir / "feature_importance.png"
    plot_feature_importance(
        model=model,
        feature_names=feature_names,
        output_path=out3,
        top_n=15,
    )
    saved_paths.append(str(out3.resolve()))

    # ------------------------------------------------------------------
    # Step 4 - SHAP summary (influence of weather / calendar factors)
    # ------------------------------------------------------------------
    with engine.connect() as conn:
        sample = pd.read_sql(text(_SQL_FEATURES), conn)
    out5 = figures_dir / "shap_summary.png"
    try:
        plot_shap_summary(model, sample[feature_names].fillna(0), out5)
        saved_paths.append(str(out5.resolve()))
    except Exception as exc:  # SHAP is optional for the rest of the report
        logger.warning("SHAP plot skipped: %s", exc)

    logger.info("All plots generated: %s", saved_paths)
    return saved_paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    # Add project root to sys.path so config / src are importable
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from config import settings
    from src.database.db_manager import get_engine, init_db

    _engine = get_engine(settings.DATABASE_URL)
    init_db(_engine)

    saved = generate_all_plots(
        engine=_engine,
        model_path=settings.MODEL_PATH,
        figures_dir=settings.FIGURES_DIR,
    )

    print("\n✅ Plots saved:")
    for p in saved:
        print(f"   {p}")

