# ✈️ Flight Arrival Delay Prediction


## 📌 Project Overview

Flight delays cost airlines and passengers significant time and money, especially at major hubs like Dallas/Fort Worth International Airport (DFW), where even small disruptions can cascade across thousands of downstream flights.

This project builds a **two-stage machine learning framework** to predict flight arrival delays by combining real-time operational flight data (ETD/ETA updates) with historical weather conditions at DFW.

1. **Classification** — Will the flight be delayed or arrive on time?
2. **Regression** — If delayed, how many minutes will it be delayed by?

## 🎯 Research Questions

- **RQ1:** Can we predict whether a flight will be delayed before takeoff using ETD signals and weather data?
- **RQ2:** Can we predict the magnitude (in minutes) of an arrival delay?
- **RQ3:** Which weather variables most strongly impact delay occurrence and duration?
- **RQ4:** Do the models generalize to unseen future flight data?

## 📊 Data

- **ETA Dataset:** 2.7M+ raw operational records (ETD updates, CYCLE records), filtered down to **~1.12 million unique flights**
- **Weather Dataset:** Hourly weather data from the Open-Meteo API (temperature, wind speed, precipitation, humidity, pressure) at DFW
- **Features:** 14+ engineered variables across temporal, operational, and environmental categories

## 🛠️ Tools & Libraries

- **Python**
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — visualization
- **Scikit-learn** — preprocessing and baseline models
- **XGBoost, LightGBM** — final classification and regression models
- **SHAP** — model interpretability

## 🧪 Methodology

1. Integrated ETA operational data with historical weather data
2. Cleaned and filtered ~2.7M raw records down to ~1.12M unique flight records
3. Engineered temporal, operational, and weather-based features
4. Performed exploratory data analysis (missing data, outliers, correlations, feature distributions)
5. Trained classification (XGBoost) and regression (XGBoost/LightGBM) models
6. Tuned hyperparameters using `RandomizedSearchCV` with 3-fold cross-validation
7. Used a **temporal train-test-validation split** to avoid data leakage and simulate real-world deployment
8. Applied SHAP analysis for feature interpretability

## 📈 Results

### Classification Model (XGBoost) — Delay vs. On-Time

| Model | Accuracy | Recall (Delayed) | F1 (Delayed) | AUC-ROC |
|---|---|---|---|---|
| XGBoost (Tuned) | 0.86 | 0.69 | 0.7844 | 0.8705 |
| XGBoost (Default) | 0.8556 | 0.6958 | 0.7851 | 0.8745 |

### Regression Model — Delay Duration (minutes)

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| XGBoost (Tuned) | 9.73 | 25.60 | 0.7304 |
| XGBoost (Default) | 9.719 | 25.580 | 0.731 |

**Key findings:**
- Wind speed, wind gusts, and precipitation are strong predictors of both delay occurrence and severity
- The classifier achieves an AUC-ROC of ~0.87, reliably flagging delayed flights (91% precision)
- The regression model explains ~73% of the variance in delay duration (R² ≈ 0.73)
- Temporal validation confirms the models generalize well to unseen future flights

## ⚠️ Limitations & Future Work

- Weather data was daily-level rather than hourly/real-time at exact departure coordinates
- Models would need frequent retraining in production to capture seasonal/trend shifts
- Significant unexplained variance remains in delay duration prediction
- Future work could incorporate route-specific weather, aircraft equipment type, and network-wide flight status, plus probabilistic forecasting instead of point estimates


## 📁 Repository Contents

- `flight_file.ipynb` — Full pipeline: data merging, cleaning, feature engineering, weather enrichment (Open-Meteo API), EDA, model training/tuning, and evaluation
- `requirements.txt` — Python libraries needed to run the notebook (`pip install -r requirements.txt`)

## 📂 Data

The datasets used (ETA operational records and airport metadata) are not included in this repository due to size and licensing. To run the notebook:
1. Obtain the ETA and Airport info datasets
2. Update the file paths at the top of the notebook to point to your local copies
3. Weather data is fetched automatically via the free [Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api) (no API key required)
