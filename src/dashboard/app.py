"""
src/dashboard/app.py
====================
Streamlit dashboard for interactive visualisation of energy price predictions.

The dashboard is organised into three tabs:

1. **Predikce vs. Skutečnost** – Interactive comparison of actual and predicted
   prices with key metric cards and a worst-error table.
2. **Předpověď na zítřek**    – Bar chart of tomorrow's hourly price forecast
   with cheapest / most expensive hour highlights.
3. **Analýza faktorů (SHAP)** – SHAP summary plot and textual explanation of
   the top-3 most impactful model features.

Run with::

    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Path bootstrap – make project root importable regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from src.database.db_manager import get_engine, init_db
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="⚡ Predikce cen elektřiny",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Shared engine  (created once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_engine():
    """Return a cached SQLAlchemy engine for the project database.

    Returns:
        A SQLAlchemy ``Engine`` instance, reused across Streamlit re-runs.
    """
    engine = get_engine(settings.DATABASE_URL)
    init_db(engine)
    return engine


ENGINE = _get_engine()


# ---------------------------------------------------------------------------
# Database loading helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_actuals_and_predictions(
    start_date: date,
    end_date: date,
    model_version: str,
) -> pd.DataFrame:
    """Load actuals and predictions joined on ``timestamp_utc``.

    Results are cached for 5 minutes to avoid hammering the database on
    every Streamlit interaction.

    Args:
        start_date: Inclusive start date (UTC) for the query.
        end_date: Inclusive end date (UTC) for the query.
        model_version: Model version string to filter ``model_predictions``.

    Returns:
        DataFrame with columns ``["timestamp_utc", "actual", "predicted"]``
        sorted by ``timestamp_utc``.  Empty DataFrame if no data found.
    """
    sql = text("""
        SELECT
            f.timestamp_utc,
            f.price_eur_mwh          AS actual,
            p.predicted_price_eur    AS predicted
        FROM fact_hourly_market_features f
        JOIN model_predictions p
            ON f.timestamp_utc = p.target_timestamp_utc
        WHERE f.price_eur_mwh IS NOT NULL
          AND p.model_version = :model_version
          AND f.timestamp_utc >= :start_date
          AND f.timestamp_utc <= :end_date
        ORDER BY f.timestamp_utc
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "model_version": model_version,
                "start_date": start_date.isoformat(),
                "end_date": f"{end_date.isoformat()}T23:59:59",
            },
        )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_available_model_versions() -> list[str]:
    """Fetch distinct model version strings from ``model_predictions``.

    Returns:
        Sorted list of model version strings present in the database, or an
        empty list if the table is empty or the query fails.
    """
    sql = text("SELECT DISTINCT model_version FROM model_predictions ORDER BY model_version DESC")
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(sql, conn)
        return df["model_version"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_next_day_predictions(model_version: str) -> pd.DataFrame:
    """Load next-day predicted prices from ``model_predictions``.

    Args:
        model_version: Model version string to filter.

    Returns:
        DataFrame with columns ``["timestamp_utc", "predicted_price_eur"]``
        for tomorrow's 24 hours.  Empty DataFrame if no predictions exist.
    """
    tomorrow = date.today() + timedelta(days=1)
    start_utc = f"{tomorrow.isoformat()}T00:00:00"
    end_utc = f"{tomorrow.isoformat()}T23:59:59"

    sql = text("""
        SELECT
            target_timestamp_utc AS timestamp_utc,
            predicted_price_eur
        FROM model_predictions
        WHERE model_version = :model_version
          AND target_timestamp_utc >= :start_utc
          AND target_timestamp_utc <= :end_utc
        ORDER BY timestamp_utc
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "model_version": model_version,
                "start_utc": start_utc,
                "end_utc": end_utc,
            },
        )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def load_feature_sample(n_rows: int = 300) -> pd.DataFrame:
    """Load a recent sample of feature rows from ``fact_hourly_market_features``.

    Args:
        n_rows: Maximum number of rows to load.  Defaults to ``300``.

    Returns:
        DataFrame of feature rows sorted by ``timestamp_utc`` descending.
        Empty DataFrame if the table is empty.
    """
    sql = text(f"""
        SELECT *
        FROM fact_hourly_market_features
        WHERE price_eur_mwh IS NOT NULL
        ORDER BY timestamp_utc DESC
        LIMIT {int(n_rows)}
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Nastavení")

# Date range picker
_today = date.today()
_default_start = _today - timedelta(days=30)
selected_start: date = st.sidebar.date_input("Datum od", value=_default_start)  # type: ignore[assignment]
selected_end: date = st.sidebar.date_input("Datum do", value=_today)  # type: ignore[assignment]

if selected_start > selected_end:
    st.sidebar.error("Datum 'od' musí být před datem 'do'.")
    st.stop()

# Model version selector
_model_versions = load_available_model_versions()
if not _model_versions:
    _model_versions = [settings.MODEL_VERSION]

selected_model_version: str = st.sidebar.selectbox(
    "Verze modelu",
    options=_model_versions,
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Data jsou načítána z projektu Energy Price Prediction.")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["📈 Predikce vs. Skutečnost", "🔮 Předpověď na zítřek", "🔬 Analýza faktorů (SHAP)"]
)

# ===========================================================================
# TAB 1 – Actual vs Predicted
# ===========================================================================
with tab1:
    st.subheader("📈 Predikce vs. Skutečnost")

    with st.spinner("Načítám data …"):
        df_main = load_actuals_and_predictions(
            start_date=selected_start,
            end_date=selected_end,
            model_version=selected_model_version,
        )

    if df_main.empty:
        st.warning(
            "⚠️ Pro zvolené období a verzi modelu nejsou k dispozici žádná data. "
            "Zkuste upravit rozsah dat nebo verzi modelu."
        )
    else:
        # --- Metric cards ---
        metrics_dict = compute_all_metrics(
            df_main["actual"].to_numpy(),
            df_main["predicted"].to_numpy(),
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "MAE (EUR/MWh)",
            f"{metrics_dict['mae']:.2f}",
            help="Průměrná absolutní chyba predikce v EUR/MWh.",
        )
        col2.metric(
            "RMSE (EUR/MWh)",
            f"{metrics_dict['rmse']:.2f}",
            help="Střední kvadratická chyba – penalizuje velké odchylky.",
        )
        col3.metric(
            "MAPE (%)",
            f"{metrics_dict['mape']:.1f}%",
            help="Průměrná absolutní procentuální chyba.",
        )
        col4.metric(
            "Směrová přesnost",
            f"{metrics_dict['directional_accuracy']:.1%}",
            help="Podíl hodin, kdy model správně odhadl směr změny ceny.",
        )

        st.markdown("---")

        # --- Interactive line chart ---
        st.markdown("**Vývoj cen – skutečnost vs. predikce**")
        chart_df = df_main.set_index("timestamp_utc")[["actual", "predicted"]].rename(
            columns={"actual": "Skutečnost (EUR/MWh)", "predicted": "Predikce (EUR/MWh)"}
        )
        st.line_chart(chart_df)

        st.markdown("---")

        # --- Worst errors table ---
        st.markdown("**Největší predikční chyby**")
        err_df = df_main.copy()
        err_df["chyba (EUR/MWh)"] = (err_df["actual"] - err_df["predicted"]).abs()
        err_df = err_df.sort_values("chyba (EUR/MWh)", ascending=False).head(20)
        err_df = err_df.rename(
            columns={
                "timestamp_utc": "Čas (UTC)",
                "actual": "Skutečnost",
                "predicted": "Predikce",
            }
        )[["Čas (UTC)", "Skutečnost", "Predikce", "chyba (EUR/MWh)"]]
        st.dataframe(
            err_df.style.format({"Skutečnost": "{:.2f}", "Predikce": "{:.2f}", "chyba (EUR/MWh)": "{:.2f}"}),
            width="stretch",
        )

# ===========================================================================
# TAB 2 – Next-Day Forecast
# ===========================================================================
with tab2:
    st.subheader("🔮 Předpověď na zítřek")

    with st.spinner("Načítám předpověď …"):
        next_day_df = load_next_day_predictions(selected_model_version)

    if next_day_df.empty:
        st.warning(
            "⚠️ Pro zítřejší den nejsou k dispozici žádné predikce. "
            "Spusťte predikční pipeline nebo zkuste jinou verzi modelu."
        )
    else:
        prices = next_day_df["predicted_price_eur"].to_numpy()
        p20 = float(np.percentile(prices, 20))
        p80 = float(np.percentile(prices, 80))

        # Bar chart coloured by percentile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        next_day_df["hour"] = next_day_df["timestamp_utc"].dt.hour
        hours = next_day_df["hour"].tolist()
        labels = [f"{h:02d}:00" for h in hours]

        colours = [
            "firebrick" if p > p80 else ("seagreen" if p < p20 else "slategray")
            for p in prices
        ]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(range(len(prices)), prices, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(prices)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Predikovaná cena (EUR/MWh)")
        ax.set_title(
            f"Hodinová předpověď cen – {date.today() + timedelta(days=1)}\n"
            f"🟢 Levné (< {p20:.1f})   🔴 Drahé (> {p80:.1f})",
            fontsize=11,
        )
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")

        # Cheapest and most expensive hours
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🟢 Nejlevnějších 6 hodin**")
            cheap = next_day_df.nsmallest(6, "predicted_price_eur")[
                ["hour", "predicted_price_eur"]
            ].rename(columns={"hour": "Hodina", "predicted_price_eur": "Cena (EUR/MWh)"})
            st.dataframe(cheap.style.format({"Cena (EUR/MWh)": "{:.2f}"}), width="stretch")

        with col_b:
            st.markdown("**🔴 Nejdražších 6 hodin**")
            expensive = next_day_df.nlargest(6, "predicted_price_eur")[
                ["hour", "predicted_price_eur"]
            ].rename(columns={"hour": "Hodina", "predicted_price_eur": "Cena (EUR/MWh)"})
            st.dataframe(expensive.style.format({"Cena (EUR/MWh)": "{:.2f}"}), width="stretch")

# ===========================================================================
# TAB 3 – SHAP Feature Analysis
# ===========================================================================
with tab3:
    st.subheader("🔬 Analýza faktorů (SHAP)")

    model_path = Path(settings.MODEL_PATH)

    if not model_path.exists():
        st.warning(
            f"⚠️ Soubor modelu nebyl nalezen: `{model_path}`. "
            "Spusťte nejprve trénink modelu."
        )
    else:
        with st.spinner("Načítám model a počítám SHAP hodnoty …"):
            try:
                import joblib
                import shap
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                artefact = joblib.load(model_path)
                # ModelTrainer.save_model stores {"model", "feature_cols", ...}
                model = artefact["model"] if isinstance(artefact, dict) else artefact
                feat_df = load_feature_sample(n_rows=300)

                if isinstance(artefact, dict) and artefact.get("feature_cols"):
                    feature_cols = list(artefact["feature_cols"])
                else:
                    non_feature_cols = {"timestamp_utc", "price_eur_mwh", "updated_at"}
                    feature_cols = [c for c in feat_df.columns if c not in non_feature_cols]
                X_sample = feat_df[feature_cols].fillna(0)

                if X_sample.empty:
                    st.warning("⚠️ Žádná dostupná data příznaků pro výpočet SHAP.")
                else:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                    # SHAP summary bar plot
                    st.markdown("**SHAP – průměrný absolutní vliv příznaků**")
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values,
                        X_sample,
                        plot_type="bar",
                        show=False,
                        max_display=15,
                    )
                    st.pyplot(plt.gcf())
                    plt.close("all")

                    # --- Top-3 text explanation ---
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
                    top3_features = [feature_cols[i] for i in top3_idx]
                    top3_values = [mean_abs_shap[i] for i in top3_idx]

                    st.markdown("---")
                    st.markdown("**🏆 Top 3 nejdůležitější faktory**")
                    for rank, (feat, val) in enumerate(zip(top3_features, top3_values), start=1):
                        st.markdown(
                            f"{rank}. **`{feat}`** – průměrný absolutní vliv na cenu: "
                            f"**{val:.3f} EUR/MWh**"
                        )

            except ImportError as exc:
                st.error(
                    f"⚠️ Chybí závislost pro SHAP analýzu: `{exc}`. "
                    "Nainstalujte ji pomocí `pip install shap joblib`."
                )
            except Exception as exc:
                logger.exception("SHAP analysis failed.")
                st.error(f"❌ Chyba při výpočtu SHAP hodnot: {exc}")

