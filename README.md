# flight-delay-prediction
Two-stage ML framework (XGBoost/LightGBM) predicting flight arrival delays at DFW using operational and weather data
✈️ Flight Arrival Delay Prediction

ADTA 5940 Analytics Capstone — University of North Texas
Group 8 | Spring 2026

📌 Project Overview

Flight delays cost airlines and passengers significant time and money, especially at major hubs like Dallas/Fort Worth International Airport (DFW), where even small disruptions can cascade across thousands of downstream flights.

This project builds a two-stage machine learning framework to predict flight arrival delays by combining real-time operational flight data (ETD/ETA updates) with historical weather conditions at DFW.


Classification — Will the flight be delayed or arrive on time?
Regression — If delayed, how many minutes will it be delayed by?


🎯 Research Questions


RQ1: Can we predict whether a flight will be delayed before takeoff using ETD signals and weather data?
RQ2: Can we predict the magnitude (in minutes) of an arrival delay?
RQ3: Which weather variables most strongly impact delay occurrence and duration?
RQ4: Do the models generalize to unseen future flight data?


📊 Data


ETA Dataset: 2.7M+ raw operational records (ETD updates, CYCLE records), filtered down to ~1.12 million unique flights
Weather Dataset: Hourly weather data from the Open-Meteo API (temperature, wind speed, precipitation, humidity, pressure) at DFW
Features: 14+ engineered variables across temporal, operational, and environmental categories


🛠️ Tools & Libraries


Python
Pandas, NumPy — data manipulation
Matplotlib, Seaborn — visualization
Scikit-learn — preprocessing and baseline models
XGBoost, LightGBM — final classification and regression models
SHAP — model interpretability


🧪 Methodology


Integrated ETA operational data with historical weather data
Cleaned and filtered ~2.7M raw records down to ~1.12M unique flight records
Engineered temporal, operational, and weather-based features
Performed exploratory data analysis (missing data, outliers, correlations, feature distributions)
Trained classification (XGBoost) and regression (XGBoost/LightGBM) models
Tuned hyperparameters using RandomizedSearchCV with 3-fold cross-validation
Used a temporal train-test-validation split to avoid data leakage and simulate real-world deployment
Applied SHAP analysis for feature interpretability


📈 Results

Classification Model (XGBoost) — Delay vs. On-Time

ModelAccuracyRecall (Delayed)F1 (Delayed)AUC-ROCXGBoost (Tuned)0.860.690.78440.8705XGBoost (Default)0.85560.69580.78510.8745

Regression Model — Delay Duration (minutes)

ModelMAE (min)RMSE (min)R²XGBoost (Tuned)9.7325.600.7304XGBoost (Default)9.71925.5800.731

Key findings:


Wind speed, wind gusts, and precipitation are strong predictors of both delay occurrence and severity
The classifier achieves an AUC-ROC of ~0.87, reliably flagging delayed flights (91% precision)
The regression model explains ~73% of the variance in delay duration (R² ≈ 0.73)
Temporal validation confirms the models generalize well to unseen future flights


⚠️ Limitations & Future Work


Weather data was daily-level rather than hourly/real-time at exact departure coordinates
Models would need frequent retraining in production to capture seasonal/trend shifts
Significant unexplained variance remains in delay duration prediction
Future work could incorporate route-specific weather, aircraft equipment type, and network-wide flight status, plus probabilistic forecasting instead of point estimates


Course: ADTA 5940 — Analytics Capstone, Dr. Jingjing Tong

📁 Repository Contents


FinalReport_Group8.docx — Full capstone report (literature review, methodology, EDA, modeling, results, conclusion)
FlightDelayPrediction_Group8.pptx — Project presentation slides
(add your notebooks/scripts here, e.g. notebooks/, src/)
