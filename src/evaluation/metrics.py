"""
src/evaluation/metrics.py
=========================
Evaluation metrics for energy price prediction models.

This module provides standard regression and domain-specific metrics used to
assess the quality of hourly electricity price forecasts.  All functions accept
both NumPy arrays and Pandas Series so they integrate naturally with the rest of
the pipeline.

Typical usage::

    from src.evaluation import metrics

    mae  = metrics.mae(actuals, predictions)
    rmse = metrics.rmse(actuals, predictions)
    mape = metrics.mape(actuals, predictions)
    da   = metrics.directional_accuracy(actuals, predictions)
    all_ = metrics.compute_all_metrics(actuals, predictions)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_array(x: np.ndarray | pd.Series) -> np.ndarray:
    """Convert *x* to a 1-D float64 NumPy array.

    Args:
        x: Input data as a NumPy array or Pandas Series.

    Returns:
        A 1-D ``np.ndarray`` of dtype ``float64``.
    """
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def mae(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Compute Mean Absolute Error (MAE).

    .. math::
        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|

    Args:
        actual: Array of observed (ground-truth) values with shape ``(n,)``.
        predicted: Array of model predictions with shape ``(n,)``.

    Returns:
        The MAE as a Python ``float``, expressed in the same unit as the inputs
        (EUR/MWh for this project).

    Raises:
        ValueError: If *actual* and *predicted* have different lengths.

    Example::

        >>> mae(np.array([10, 20, 30]), np.array([12, 18, 33]))
        2.3333333333333335
    """
    a = _to_array(actual)
    p = _to_array(predicted)
    if a.shape != p.shape:
        raise ValueError(
            f"Shape mismatch: actual={a.shape}, predicted={p.shape}"
        )
    return float(np.mean(np.abs(a - p)))


def rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Compute Root Mean Squared Error (RMSE).

    RMSE penalises large errors more heavily than MAE, making it sensitive to
    outlier predictions – important in electricity markets where price spikes
    carry significant financial risk.

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}

    Args:
        actual: Array of observed values with shape ``(n,)``.
        predicted: Array of model predictions with shape ``(n,)``.

    Returns:
        The RMSE as a Python ``float`` in the same unit as the inputs
        (EUR/MWh).

    Raises:
        ValueError: If *actual* and *predicted* have different lengths.

    Example::

        >>> rmse(np.array([10, 20, 30]), np.array([12, 18, 33]))
        2.6457513110645907
    """
    a = _to_array(actual)
    p = _to_array(predicted)
    if a.shape != p.shape:
        raise ValueError(
            f"Shape mismatch: actual={a.shape}, predicted={p.shape}"
        )
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    epsilon: float = 1e-6,
) -> float:
    """Compute Mean Absolute Percentage Error (MAPE).

    Near-zero actual prices (common during negative-price hours in the Czech
    market) are guarded by *epsilon* to prevent division by zero or
    astronomically large percentage errors.

    .. math::
        \\text{MAPE} = \\frac{100}{n} \\sum_{i=1}^{n}
            \\frac{|y_i - \\hat{y}_i|}{\\max(|y_i|,\\, \\varepsilon)}

    Args:
        actual: Array of observed values with shape ``(n,)``.
        predicted: Array of model predictions with shape ``(n,)``.
        epsilon: Small positive constant added to the denominator to avoid
            division by zero when actual prices are near zero.
            Defaults to ``1e-6``.

    Returns:
        MAPE expressed as a percentage (e.g. ``5.3`` means 5.3 %).

    Raises:
        ValueError: If *actual* and *predicted* have different lengths.

    Example::

        >>> mape(np.array([10, 20, 30]), np.array([12, 18, 33]))
        11.11111111...
    """
    a = _to_array(actual)
    p = _to_array(predicted)
    if a.shape != p.shape:
        raise ValueError(
            f"Shape mismatch: actual={a.shape}, predicted={p.shape}"
        )
    denominator = np.maximum(np.abs(a), epsilon)
    return float(np.mean(np.abs(a - p) / denominator) * 100.0)


def directional_accuracy(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> float:
    """Compute Directional Accuracy (DA).

    Measures the fraction of consecutive hours in which the model correctly
    predicts whether the price *rose* or *fell* relative to the previous hour.
    A value of 0.50 corresponds to random guessing; higher values indicate the
    model captures price-movement direction.

    Algorithm:
        1. Compute first-differences with ``np.diff``.
        2. Extract the sign of each difference (``np.sign``).
        3. Count how often ``sign(Δy_i) == sign(Δŷ_i)``.

    .. math::
        \\text{DA} = \\frac{1}{n-1}
            \\sum_{i=2}^{n} \\mathbf{1}[\\text{sign}(y_i - y_{i-1})
            = \\text{sign}(\\hat{y}_i - \\hat{y}_{i-1})]

    Args:
        actual: Array of observed values with shape ``(n,)``.
        predicted: Array of model predictions with shape ``(n,)``.

    Returns:
        Directional accuracy as a fraction in ``[0, 1]``.  Returns ``float('nan')``
        when the input has fewer than two elements (no differences to compare).

    Raises:
        ValueError: If *actual* and *predicted* have different lengths.

    Example::

        >>> directional_accuracy(np.array([10, 12, 11, 14]),
        ...                      np.array([10, 13, 10, 15]))
        1.0
    """
    a = _to_array(actual)
    p = _to_array(predicted)
    if a.shape != p.shape:
        raise ValueError(
            f"Shape mismatch: actual={a.shape}, predicted={p.shape}"
        )
    if len(a) < 2:
        return float("nan")

    actual_diff = np.diff(a)
    predicted_diff = np.diff(p)

    # np.sign returns 0 for no-change hours; treat 0 vs 0 as a correct call.
    matches = np.sign(actual_diff) == np.sign(predicted_diff)
    return float(np.mean(matches))


# ---------------------------------------------------------------------------
# Composite helper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute all four evaluation metrics in a single call.

    Convenience wrapper that calls :func:`mae`, :func:`rmse`, :func:`mape`,
    and :func:`directional_accuracy` and returns their results in a dictionary
    suitable for logging or storing in the ``evaluation_results`` table.

    Args:
        actual: Array of observed values with shape ``(n,)``.
        predicted: Array of model predictions with shape ``(n,)``.

    Returns:
        A dictionary with the following keys:

        - ``"mae"``                  – Mean Absolute Error (EUR/MWh).
        - ``"rmse"``                 – Root Mean Squared Error (EUR/MWh).
        - ``"mape"``                 – Mean Absolute Percentage Error (%).
        - ``"directional_accuracy"`` – Directional accuracy in ``[0, 1]``.
        - ``"n_samples"``            – Number of data points evaluated.

    Example::

        >>> result = compute_all_metrics(actuals_array, predictions_array)
        >>> print(result["mae"])
        3.14
    """
    a = _to_array(actual)
    p = _to_array(predicted)

    return {
        "mae": mae(a, p),
        "rmse": rmse(a, p),
        "mape": mape(a, p),
        "directional_accuracy": directional_accuracy(a, p),
        "n_samples": int(len(a)),
    }
