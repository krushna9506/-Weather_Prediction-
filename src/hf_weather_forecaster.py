"""
Module: src/hf_weather_forecaster.py
Integrates Google Research's TimesFM 3.0 (google/timesfm-3.0-pytorch, 1.23 GB)
from Hugging Face Hub for zero-shot probabilistic weather forecasting
on real-time Open-Meteo hourly atmospheric telemetry.
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HF_MODEL_ID = "google/timesfm-3.0-pytorch"
_cached_forecaster = None


def get_timesfm_forecaster():
    """Loads and caches google/timesfm-3.0-pytorch (1.23 GB) from Hugging Face Hub."""
    global _cached_forecaster
    if _cached_forecaster is None:
        from timesfm import TimesFM3Forecaster
        _cached_forecaster = TimesFM3Forecaster.from_pretrained(HF_MODEL_ID)
    return _cached_forecaster


def fetch_past_hourly_series(lat: float, lon: float, past_days: int = 7, forecast_days: int = 2):
    """Fetches past 7 days + future reference hourly telemetry from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&past_days={past_days}&forecast_days={forecast_days}"
        f"&hourly=temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m&timezone=auto"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SkyCastAI/4.0"})
    with urllib.request.urlopen(req, timeout=6.0) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("hourly", {})


def forecast_weather_with_timesfm(lat: float, lon: float, horizon: int = 24):
    """
    Runs zero-shot probabilistic forecasting using google/timesfm-3.0-pytorch
    on live past-7-day hourly temperature & pressure sequences.
    """
    hourly = fetch_past_hourly_series(lat, lon, past_days=7, forecast_days=2)
    times = hourly.get("time", [])
    temps = np.array(hourly.get("temperature_2m", []), dtype=np.float32)
    pressures = np.array(hourly.get("surface_pressure", []), dtype=np.float32)

    if len(temps) < 48:
        raise ValueError("Insufficient hourly telemetry history from Open-Meteo.")

    # Use past 168 hours (7 days) up to current hour as model context
    ctx_len = min(168, len(temps) - horizon)
    temp_ctx = temps[:ctx_len]
    slp_ctx = pressures[:ctx_len]
    ref_temps = temps[ctx_len:ctx_len + horizon]
    future_times = times[ctx_len:ctx_len + horizon]

    forecaster = get_timesfm_forecaster()

    temp_out = forecaster.predict(context=temp_ctx, horizon=horizon, return_quantiles=True)
    slp_out = forecaster.predict(context=slp_ctx, horizon=horizon, return_quantiles=False)

    temp_med = np.array(temp_out.forecast).flatten()[:horizon]
    slp_med = np.array(slp_out.forecast).flatten()[:horizon]

    # Extract 10th and 90th percentile uncertainty bands if available
    q_arr = np.array(temp_out.quantiles) if getattr(temp_out, "quantiles", None) is not None else None
    if q_arr is not None and q_arr.ndim >= 2:
        q_2d = q_arr.reshape(-1, q_arr.shape[-1])
        temp_low = q_2d[:horizon, 0]
        temp_high = q_2d[:horizon, -1]
    else:
        temp_low = temp_med - 1.5
        temp_high = temp_med + 1.5

    df_forecast = pd.DataFrame({
        "Time": pd.to_datetime(future_times),
        "TimesFM 3.0 Forecast (°C)": np.round(temp_med, 2),
        "Lower 10% (°C)": np.round(temp_low, 2),
        "Upper 90% (°C)": np.round(temp_high, 2),
        "Open-Meteo NWP (°C)": np.round(ref_temps[:len(future_times)], 2),
        "TimesFM Pressure (hPa)": np.round(slp_med, 1),
    })

    return {
        "model_id": HF_MODEL_ID,
        "device": str(forecaster.device),
        "context_hours": int(ctx_len),
        "horizon_hours": int(horizon),
        "forecast_df": df_forecast,
    }
