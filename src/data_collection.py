"""
Module: data_collection.py
Stage A: Data Collection and Attribute Characterization [2 Marks - CO1, L3]
Aligned with the WEATHER-5K Benchmark (taohan10200/WEATHER-5K)
Covers 5,672 Global Station Profiles, Psychrometric Dewpoint, Wind Direction (0-360°), and Multi-Horizon Forecasts
"""

import os
import pandas as pd
from typing import Dict, Tuple, Any

def get_attribute_metadata() -> pd.DataFrame:
    """
    Returns comprehensive domain metadata for all attributes in the WEATHER-5K Global Station Dataset.
    Specifies data types, measurement units, physical sensor sources, ranges, and domain roles.
    """
    attributes = [
        {
            "Attribute": "Station_ID",
            "Data Type": "Categorical (Identifier)",
            "Unit": "WMO Station Code",
            "Sensor / Source": "World Meteorological Organization (WMO) Global Station Index",
            "Valid Range": "e.g. US722190, CH066600, UK037720",
            "Domain Significance": "Unique spatial node identifier among 5,672 worldwide weather stations in WEATHER-5K."
        },
        {
            "Attribute": "Station_Name",
            "Data Type": "Text (Metadata)",
            "Unit": "Station Name",
            "Sensor / Source": "NOAA / Global Station Weather Forecasting Directory",
            "Valid Range": "e.g. London Heathrow, Tokyo Haneda, Zermatt Matterhorn",
            "Domain Significance": "Human-readable geographical observatory location name."
        },
        {
            "Attribute": "Latitude",
            "Data Type": "Float (Continuous)",
            "Unit": "°N / °S",
            "Sensor / Source": "Geodetic WGS84 GPS Telemetry",
            "Valid Range": "[-60.0, 75.0]",
            "Domain Significance": "Primary solar declination determinant; modulates solar zenith angle and thermal insolation."
        },
        {
            "Attribute": "Longitude",
            "Data Type": "Float (Continuous)",
            "Unit": "°E / °W",
            "Sensor / Source": "Geodetic WGS84 GPS Telemetry",
            "Valid Range": "[-180.0, 180.0]",
            "Domain Significance": "Zonal geographic position; dictates diurnal solar cycle timing and maritime advection."
        },
        {
            "Attribute": "Altitude (m)",
            "Data Type": "Float (Continuous)",
            "Unit": "meters (m)",
            "Sensor / Source": "Digital Elevation Model (DEM) / Altimeter",
            "Valid Range": "[0.0, 3500.0]",
            "Domain Significance": "Station elevation above sea level; dictates barometric pressure decay and lapse rate cooling."
        },
        {
            "Attribute": "Region",
            "Data Type": "Categorical (Nominal)",
            "Unit": "Climate Zone",
            "Sensor / Source": "Köppen-Geiger Global Climate Classification",
            "Valid Range": "{Equatorial Tropical, Temperate Continental, Arid / Desert, Mediterranean Maritime, Nordic / Sub-Polar, Alpine Mountain Highlands}",
            "Domain Significance": "Macro-scale geographic bioregion capturing prevailing air masses and moisture dynamics."
        },
        {
            "Attribute": "Location",
            "Data Type": "Categorical (Nominal)",
            "Unit": "Topography / Terrain",
            "Sensor / Source": "GIS Topographical Tag",
            "Valid Range": "{coastal, inland, mountain}",
            "Domain Significance": "Local terrain topography governing maritime thermal inertia, coastal breeze, and orographic lift."
        },
        {
            "Attribute": "Season",
            "Data Type": "Categorical (Nominal)",
            "Unit": "Temporal Season",
            "Sensor / Source": "Astronomical Solar Calendar Metadata",
            "Valid Range": "{Winter, Spring, Summer, Autumn}",
            "Domain Significance": "Cyclic climatic season capturing axial tilt insolation and daylight duration."
        },
        {
            "Attribute": "Temperature (TMP)",
            "Data Type": "Float (Continuous)",
            "Unit": "°C",
            "Sensor / Source": "Thermistor / Platinum RTD (PT100)",
            "Valid Range": "[-35.0, 50.0]",
            "Domain Significance": "WEATHER-5K core telemetry element; atmospheric kinetic thermal energy (Dataset Mean: 12.71°C)."
        },
        {
            "Attribute": "Dewpoint (DPT)",
            "Data Type": "Float (Continuous)",
            "Unit": "°C",
            "Sensor / Source": "Chilled-Mirror Hygrometer",
            "Valid Range": "[-40.0, 35.0]",
            "Domain Significance": "WEATHER-5K core telemetry element; temperature at which air reaches water vapor saturation (Mean: 6.53°C)."
        },
        {
            "Attribute": "Humidity (RH)",
            "Data Type": "Integer (Discrete)",
            "Unit": "% (RH)",
            "Sensor / Source": "Psychrometric Magnus-Tetens Formulation from TMP & DPT",
            "Valid Range": "[10, 100]",
            "Domain Significance": "Relative moisture saturation; fundamental indicator for cloud condensation and precipitation."
        },
        {
            "Attribute": "Wind Speed (WSPD)",
            "Data Type": "Float (Continuous)",
            "Unit": "km/h",
            "Sensor / Source": "Ultrasonic Anemometer",
            "Valid Range": "[0.0, 50.0]",
            "Domain Significance": "WEATHER-5K core telemetry element; horizontal air velocity (Dataset Mean: 3.37 m/s ≈ 12.1 km/h)."
        },
        {
            "Attribute": "Wind Direction (WDIR)",
            "Data Type": "Float (Continuous)",
            "Unit": "Degrees (0-360°)",
            "Sensor / Source": "Potentiometric Wind Vane",
            "Valid Range": "[0.0, 360.0]",
            "Domain Significance": "WEATHER-5K core telemetry element; compass azimuth from which wind originates (Mean: 191.19°)."
        },
        {
            "Attribute": "Precipitation (%)",
            "Data Type": "Float (Continuous)",
            "Unit": "%",
            "Sensor / Source": "Optical Rain / Snow Disdrometer",
            "Valid Range": "[0.0, 100.0]",
            "Domain Significance": "Precipitation probability and accumulation rate."
        },
        {
            "Attribute": "Cloud Cover",
            "Data Type": "Categorical (Nominal)",
            "Unit": "Oktas / Classes",
            "Sensor / Source": "Laser Ceilometer / Satellite Optical Imagery",
            "Valid Range": "{clear, partly cloudy, cloudy, overcast}",
            "Domain Significance": "Fraction of celestial dome obscured by clouds; modulates solar irradiance."
        },
        {
            "Attribute": "Atmospheric Pressure (SLP)",
            "Data Type": "Float (Continuous)",
            "Unit": "hPa (mbar)",
            "Sensor / Source": "Silicon Piezoresistive Barometer",
            "Valid Range": "[750.0, 1050.0]",
            "Domain Significance": "WEATHER-5K core element; sea-level / local barometric pressure (Mean: 1014.85 hPa)."
        },
        {
            "Attribute": "UV Index",
            "Data Type": "Integer (Discrete)",
            "Unit": "Index (0-14)",
            "Sensor / Source": "Broadband Pyranometer / UV Radiometer",
            "Valid Range": "[0, 14]",
            "Domain Significance": "Solar ultraviolet radiation intensity governed by latitude and cloud opacity."
        },
        {
            "Attribute": "Visibility (km)",
            "Data Type": "Float (Continuous)",
            "Unit": "km",
            "Sensor / Source": "Forward-Scatter Optical Transmissometer",
            "Valid Range": "[0.0, 20.0]",
            "Domain Significance": "Atmospheric transparency; attenuated by fog, rain droplets, and falling snow."
        },
        {
            "Attribute": "Forecast_24h_TMP",
            "Data Type": "Float (Continuous)",
            "Unit": "°C",
            "Sensor / Source": "WEATHER-5K Multi-Horizon Forecast Horizon (t + 24h)",
            "Valid Range": "[-35.0, 50.0]",
            "Domain Significance": "Target 24-hour lead-time predicted temperature benchmark."
        },
        {
            "Attribute": "Forecast_72h_TMP",
            "Data Type": "Float (Continuous)",
            "Unit": "°C",
            "Sensor / Source": "WEATHER-5K Multi-Horizon Forecast Horizon (t + 72h)",
            "Valid Range": "[-35.0, 50.0]",
            "Domain Significance": "Target 72-hour lead-time predicted temperature benchmark."
        },
        {
            "Attribute": "Weather Type (Target)",
            "Data Type": "Categorical (Multi-Class)",
            "Unit": "Classification Label",
            "Sensor / Source": "Official Meteorological Station Ground Truth",
            "Valid Range": "{Cloudy, Rainy, Snowy, Sunny}",
            "Domain Significance": "Ground-truth target variable representing the macro weather pattern to be classified."
        }
    ]
    return pd.DataFrame(attributes)

def load_dataset(filepath: str = "data/raw/weather_raw_data.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not os.path.exists(filepath):
        filepath = "weather_raw_data.csv"
        
    df = pd.read_csv(filepath)
    specs = {
        "Total Observations": len(df),
        "Total Attributes": df.shape[1],
        "Numerical Features": list(df.select_dtypes(include=['float64', 'int64']).columns),
        "Categorical Features": list(df.select_dtypes(include=['object']).columns),
        "Target Variable": "Weather Type",
        "Target Classes": list(df["Weather Type"].unique()),
        "Class Distribution": df["Weather Type"].value_counts().to_dict(),
        "Unique Stations": df["Station_Name"].nunique() if "Station_Name" in df.columns else 0,
        "Geographic Regions": list(df["Region"].unique()) if "Region" in df.columns else []
    }
    return df, specs

if __name__ == "__main__":
    df, specs = load_dataset()
    print("=== Section A: Data Collection Specifications (WEATHER-5K Aligned) ===")
    for k, v in specs.items():
        print(f"{k}: {v}")
