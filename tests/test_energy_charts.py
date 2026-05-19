"""Tests for the token-free Energy-Charts generation client."""

from __future__ import annotations

from src.ingestion.energy_charts_client import EnergyChartsClient


def test_parse_extracts_solar_and_wind_hourly() -> None:
    """15-minute series are averaged to hours; other series are ignored."""
    start = 1_792_447_200  # 2026-10-19 22:00 UTC
    payload = {
        "unix_seconds": [start + 900 * i for i in range(8)],  # two hours
        "production_types": [
            {"name": "Solar", "data": [0, 100, 200, 300, 400, 400, 400, 400]},
            {"name": "Wind onshore", "data": [10] * 8},
            {"name": "Nuclear", "data": [3000] * 8},
        ],
    }
    df = EnergyChartsClient._parse(payload)
    assert set(df["production_type"]) == {"solar", "wind_onshore"}
    assert len(df) == 4
    solar = df[df["production_type"] == "solar"]["actual_power_mw"].tolist()
    assert solar == [150.0, 400.0]


def test_parse_empty_payload() -> None:
    """A payload without relevant series yields an empty frame."""
    payload = {"unix_seconds": [1], "production_types": [{"name": "Nuclear", "data": [1]}]}
    assert EnergyChartsClient._parse(payload).empty
