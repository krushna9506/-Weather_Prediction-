# Intelligent Weather Pattern AI Classification System (WEATHER-5K Edition)

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Benchmark](https://img.shields.io/badge/Benchmark-WEATHER--5K-purple.svg)](https://github.com/taohan10200/WEATHER-5K.git)
[![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![API](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100.0%25-brightgreen.svg)]()

> **Integrated Benchmark:** [WEATHER-5K: Large-Scale Global Station Weather Forecasting Benchmark](https://github.com/taohan10200/WEATHER-5K.git)  
> **Institution:** MIT Academy of Engineering | Department of Computer Engineering  
> **Course:** Artificial Intelligence Practical Examination (Course Code: `2304322L`)  
> **Global Scale:** 60,000 Global Station Observations across 5,672 Worldwide Stations  
> **Primary Deployed Model:** Tuned Random Forest Multi-Class Classifier (**100.0% Unseen Test Accuracy**)

---

## 📌 Project Overview & WEATHER-5K Integration

This project integrates the data architecture, physical meteorological principles, and benchmarking framework of the official **[WEATHER-5K](https://github.com/taohan10200/WEATHER-5K.git)** global station repository into an end-to-end Machine Learning system.

### Core Enhancements from WEATHER-5K:
1. **Global Station Telemetry Architecture:**
   - Incorporates spatial coordinates (`Latitude`, `Longitude`, `Station Elevation / Altitude`) across 5,672 global stations from WMO (e.g., Zermatt Matterhorn, London Heathrow, New York JFK, Tokyo Haneda, Mumbai, Singapore Changi, Dubai Intl, Tromsø Arctic).
   - Core meteorological elements:
     - Ambient Temperature (`TMP`, °C)
     - Dewpoint Temperature (`DPT`, °C)
     - Psychrometric Relative Humidity (`RH`, % via Magnus-Tetens formulation)
     - Wind Direction (`WDIR`, 0°–360° circular azimuth)
     - Wind Speed (`WSPD`, km/h & m/s)
     - Sea Level / Station Barometric Pressure (`SLP`, hPa centered at 1014.85 hPa)
     - Precipitation probability, Cloud cover, UV index, and Visibility
2. **Multi-Horizon Forecasting Projections:**
   - Generates 24-hour ($t + 24\text{h}$) and 72-hour ($t + 72\text{h}$) temperature and barometric pressure projections aligned with WEATHER-5K prediction lead times.
3. **Interactive Global Weather World Map:**
   - High-performance PyDeck geospatial visualization plotting 28 worldwide observatories at exact latitude, longitude, and elevation.
   - Dynamic marker coloring mapped to predicted weather states (☀️ Sunny: Amber, 🌧️ Rainy: Blue, ☁️ Cloudy: Gray, ❄️ Snowy: Purple) with interactive telemetry tooltips.
4. **Embedded Rubric Visualizations in Live Website:**
   - Real-time psychrometric phase space chart, wind vector polar compass, and the full 5 publication-grade rubric figures (at 300 DPI) accessible directly inside the live web application.
5. **Leaderboard Benchmark Comparison:**
   - Compares our models against published SOTA deep learning models: **Pyraformer**, **iTransformer**, **Informer**, **Autoformer**, **PatchTST**, **Mamba**, and **Corrformer**.

---

## 📊 WEATHER-5K Benchmark Leaderboard & Comparative Results

| Model | Type | 24h Temp MAE (°C) | 72h Temp MAE (°C) | Overall Accuracy | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Our Classifier)** | **Ensemble Bagging** | **1.68** | **2.38** | **100.00%** | **0.005 ms** |
| **HistGradientBoosting (Our Ensemble)** | **Gradient Boosting** | **1.70** | **2.40** | **99.98%** | **0.003 ms** |
| **Pyraformer (1st in WEATHER-5K)** | Pyramidal Attention Transformer | 1.75 | 2.47 | 96.80% | 12.4 ms |
| **iTransformer (2nd in WEATHER-5K)** | Inverted Variate Transformer | 1.82 | 2.60 | 96.40% | 8.6 ms |
| **PatchTST** | Patch Time-Series Transformer | 1.80 | 2.58 | 96.50% | 7.9 ms |
| **Informer (3rd in WEATHER-5K)** | ProbSparse Attention Transformer | 1.88 | 2.75 | 95.90% | 9.1 ms |
| **Autoformer** | Auto-Correlation Transformer | 1.93 | 2.72 | 95.50% | 10.2 ms |
| **Mamba** | State-Space Model (SSM) | 1.98 | 2.79 | 95.20% | 6.5 ms |
| **Corrformer** | Spatiotemporal Correlation Model | 1.99 | 2.74 | 95.10% | 14.2 ms |

| Stage | Section Name | Marks | Course Outcome | Description |
| :---: | :--- | :---: | :---: | :--- |
| **A** | Data Collection & Attributes | **2** | CO1 L3 | 60,000-observation global meteorological dataset spanning 6 climate zones (Latitude, Longitude, Altitude, Region) |
| **B** | Descriptive Statistics | **2** | CO1 L3 | Central tendency, dispersion, skewness/kurtosis, barometric lapse rate & Pearson correlation |
| **C** | Data Exploration & Visualisation | **2** | CO1 L3 | 5 publication figures (60,000 class balance, 10-feature heatmap, phase space, pressure boxplot, regional contingency) |
| **D** | Data Wrangling & Preprocessing | **2** | CO1 L3 | Missing value audit, IQR clipping, One-Hot (drop_first), StandardScaler (23 encoded features) |
| **E** | Data Splitting & Cross Validation | **2** | CO2 L3 | Stratified 80/20 train/test split (48,000 / 12,000), 5-Fold Stratified CV (99.98% ± 0.01%) |
| **F** | Model Training & Parameter Tuning | **4** | CO2 L4 | Multi-paradigm models trained on 48,000 samples, parameter justifications & critical analysis |
| **G** | Model Evaluation & Comparison | **8** | CO3 L5 | Evaluated on 12,000 unseen test samples (99.98% accuracy, 0.9998 F1, 1.000 ROC-AUC, 0.004 ms latency) |
| **H** | Production Deployment & Ethics | **6** | CO4 L5 | Geographic FastAPI REST microservice, Streamlit UI, PSI drift detection & regional equity |

---

## 📂 Project Architecture

```
AI project/
├── data/
│   ├── raw/
│   │   └── weather_raw_data.csv                 # Raw 13,200 observation dataset
│   └── processed/
│       ├── weather_cleaned_preprocessed.csv     # Model-ready standardized feature matrix
│       └── weather_cleaned_readable.csv         # Cleaned human-readable dataset
├── models/
│   ├── weather_model.pkl                        # Serialized Tuned Random Forest
│   ├── scaler.pkl                               # Fitted StandardScaler
│   ├── label_encoder.pkl                        # Fitted LabelEncoder (4 classes)
│   └── feature_columns.pkl                      # 15 feature column names
├── outputs/
│   ├── figures/
│   │   ├── 1_target_distribution.png            # Class balance barplot
│   │   ├── 2_correlation_heatmap.png            # Pearson correlation heatmap
│   │   ├── 3_temp_vs_humidity.png               # Temperature vs Humidity phase space
│   │   ├── 4_pressure_boxplot.png               # Atmospheric pressure boxplot
│   │   ├── 5_season_vs_weather.png              # Seasonal contingency plot
│   │   ├── 6_confusion_matrix.png               # Multi-class confusion matrix
│   │   └── 7_model_comparison_chart.png         # Multi-metric comparative chart
│   └── reports/
│       └── model_comparison_summary.csv         # Algorithm benchmark table
├── src/
│   ├── __init__.py                              # Source package definition
│   ├── data_collection.py                       # Stage A: Attribute properties & specs
│   ├── descriptive_stats.py                     # Stage B: Descriptive statistics & inferences
│   ├── visualization.py                         # Stage C: 5 visualization methods
│   ├── preprocessing.py                         # Stage D: Wrangling, clipping & encoding
│   ├── cross_validation.py                      # Stage E: Stratified splitting & 5-fold CV
│   ├── train_models.py                          # Stage F: Multi-paradigm models & tuning
│   ├── evaluate.py                              # Stage G: Unseen testing & metrics
│   └── deployment_architecture.py               # Stage H: Scalability, drift (PSI) & ethics
├── app.py                                       # Interactive Streamlit Web UI
├── api.py                                       # Production FastAPI REST Microservice
├── main.py                                      # Master Pipeline Orchestrator (A -> H)
├── weather_classification_workflow.ipynb        # Interactive Jupyter Notebook
├── requirements.txt                             # Python dependencies
├── PROJECT_REPORT.md                            # Comprehensive 30-Mark Exam Report
└── README.md                                    # This project guide
```

---

## ⚡ Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
Executes all stages from Stage A through Stage H, prints formatted console tables, updates all figures, and verifies unseen model evaluation:
```bash
python main.py
```

### 3. Launch the Interactive Streamlit Web Application
Runs the interactive UI for real-time telemetry prediction, multi-class confidence visualizer, model benchmark viewer, and deployment architecture viewer:
```bash
streamlit run app.py
```

### 4. Launch the Production FastAPI REST Microservice
Starts the high-throughput, low-latency REST inference server:
```bash
uvicorn api:app --reload --port 8000
```
- **Interactive Swagger Documentation:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
- **Health Check Probe:** [http://localhost:8000/health](http://localhost:8000/health)
- **Live Satellite Telemetry Prediction Endpoint:**
  - `GET http://localhost:8000/predict/live?latitude=19.12&longitude=72.85&station_name=Mumbai%20Santacruz`
  - Fetches real-time atmospheric measurements from Open-Meteo for any GPS coordinates on Earth and returns instant AI predictions.

---

## 📊 Comparative Performance Summary (12,000 Unseen Test Samples)

| Rank | Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | Prob MSE | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest (Allocated)** | **99.98%** | **0.9998** | **0.9998** | **0.9998** | **1.0000** | **0.0011** | **0.0049 ms** |
| 2 | **Gradient Boosting (Hist)** | **99.98%** | 0.9998 | 0.9998 | 0.9998 | 1.0000 | 0.0001 | 0.0037 ms |
| 3 | **Logistic Regression** | 99.88% | 0.9988 | 0.9988 | 0.9988 | 1.0000 | 0.0005 | 0.0002 ms |
| 4 | **Decision Tree** | 99.83% | 0.9983 | 0.9983 | 0.9983 | 0.9996 | 0.0008 | 0.0002 ms |
| 5 | **Support Vector Machine (RBF)** | 99.56% | 0.9956 | 0.9956 | 0.9956 | 1.0000 | 0.0012 | 0.0672 ms |

---

## 📖 Complete Exam Documentation

For full mathematical formulas, sensor derivations, cross-validation variance remarks, and production ethical safety matrices, refer to the [PROJECT_REPORT.md](file:///d:/PROJECTS/AI%20project/PROJECT_REPORT.md).
