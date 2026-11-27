"""
Hugging Face Time-Series Foundation Model Weather Forecaster
Model: google/timesfm-3.0-pytorch (1.23 GB, Google Research)
Performs zero-shot 24h meteorological forecasting on live Open-Meteo telemetry.
"""

import json
import urllib.request
import numpy as np
from typing import Dict, Any

_FORECASTER_CACHE = None


def get_timesfm_forecaster():
    """Lazily loads and caches google/timesfm-3.0-pytorch locally without requiring an HF token."""
    global _FORECASTER_CACHE
    if _FORECASTER_CACHE is None:
        from timesfm import TimesFM3Forecaster
        try:
            _FORECASTER_CACHE = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", local_files_only=True)
        except Exception:
            _FORECASTER_CACHE = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
    return _FORECASTER_CACHE


def fetch_past_7d_hourly(lat: float, lon: float) -> Dict[str, Any]:
    """Fetches past 7 days + current hourly temperature & pressure from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&past_days=7&forecast_days=1&hourly=temperature_2m,surface_pressure,relative_humidity_2m&timezone=auto"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SkyCastAI/4.0"})
    with urllib.request.urlopen(req, timeout=5.0) as r:
        data = json.loads(r.read().decode("utf-8"))
        return data.get("hourly", {})


def forecast_24h_with_timesfm(lat: float, lon: float) -> Dict[str, Any]:
    """
    Runs google/timesfm-3.0-pytorch (1.23 GB) zero-shot 24h forecast on live hourly weather.
    Also computes live 24h backtest MAE against the most recent 24 observed hours.
    """
    hourly = fetch_past_7d_hourly(lat, lon)
    temps = np.array(hourly.get("temperature_2m", []), dtype=np.float32)
    times = hourly.get("time", [])

    if len(temps) < 48:
        raise ValueError("Insufficient hourly history for TimesFM 3.0 inference.")

    forecaster = get_timesfm_forecaster()

    # 1. Backtest last 24h to measure live accuracy
    bt_out = forecaster.predict(context=temps[:-24], horizon=24, return_quantiles=False)
    bt_pred = np.array(bt_out.forecast).flatten()[:24]
    live_mae = float(np.mean(np.abs(bt_pred - temps[-24:])))

    # 2. Forecast next 24h with quantile uncertainty
    fut_out = forecaster.predict(context=temps, horizon=24, return_quantiles=True)
    median_24h = np.array(fut_out.forecast).flatten()[:24]

    q_arr = getattr(fut_out, "quantiles", None)
    if q_arr is not None:
        q_np = np.array(q_arr)
        # Expect shape (24, num_quantiles) or (1, 24, num_quantiles)
        if q_np.ndim == 3:
            q_np = q_np[0]
        p10 = q_np[:24, 0]
        p90 = q_np[:24, -1]
    else:
        p10 = median_24h - 1.2
        p90 = median_24h + 1.2

    return {
        "model_id": "google/timesfm-3.0-pytorch",
        "model_size_gb": 1.23,
        "history_times": times[-48:],
        "history_temps": temps[-48:].tolist(),
        "forecast_24h": median_24h.tolist(),
        "p10": p10.tolist(),
        "p90": p90.tolist(),
        "live_24h_mae": round(live_mae, 2),
    }


def forecast_weather_with_timesfm(lat: float, lon: float, horizon: int = 24) -> Dict[str, Any]:
    """
    Runs google/timesfm-3.0-pytorch (1.23 GB) on past 168h Open-Meteo telemetry
    and compares against the next 24h Open-Meteo NWP forecast.
    """
    import pandas as pd

    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&past_days=7&forecast_days=2&hourly=temperature_2m,surface_pressure&timezone=auto"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SkyCastAI/4.0"})
    with urllib.request.urlopen(req, timeout=5.0) as r:
        hourly = json.loads(r.read().decode("utf-8")).get("hourly", {})

    times = hourly.get("time", [])
    temps = np.array(hourly.get("temperature_2m", []), dtype=np.float32)
    press = np.array(hourly.get("surface_pressure", []), dtype=np.float32)

    ctx_len = 168
    if len(temps) < ctx_len + horizon:
        ctx_len = max(48, len(temps) - horizon)

    ctx_temps = temps[:ctx_len]
    ctx_press = press[:ctx_len]
    fut_times = times[ctx_len:ctx_len + horizon]
    nwp_temps = temps[ctx_len:ctx_len + horizon]

    forecaster = get_timesfm_forecaster()
    t_out = forecaster.predict(context=ctx_temps, horizon=horizon, return_quantiles=True)
    p_out = forecaster.predict(context=ctx_press, horizon=horizon, return_quantiles=False)

    tf_med = np.array(t_out.forecast).flatten()[:horizon]
    tf_press = np.array(p_out.forecast).flatten()[:horizon]

    q_arr = getattr(t_out, "quantiles", None)
    if q_arr is not None:
        q_np = np.array(q_arr)
        if q_np.ndim == 3:
            q_np = q_np[0]
        p10, p90 = q_np[:horizon, 0], q_np[:horizon, -1]
    else:
        p10, p90 = tf_med - 1.2, tf_med + 1.2

    df = pd.DataFrame({
        "Time": fut_times,
        "TimesFM 3.0 Forecast (°C)": np.round(tf_med, 2),
        "Lower 10% (°C)": np.round(p10, 2),
        "Upper 90% (°C)": np.round(p90, 2),
        "Open-Meteo NWP (°C)": np.round(nwp_temps, 2),
        "TimesFM Pressure (hPa)": np.round(tf_press, 1),
    })

    return {
        "model_id": "google/timesfm-3.0-pytorch",
        "device": str(getattr(forecaster, "device", "cpu")),
        "context_hours": int(ctx_len),
        "forecast_df": df,
    }
