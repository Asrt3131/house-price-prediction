# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house prices based on property features such as living area, overall quality, year built, basement size, garage capacity, and number of bathrooms.

The model is trained on the **Kaggle House Prices (Ames) dataset** and deployed as an interactive **Streamlit** web app.

---

## 🔗 Live Demo

- **Streamlit App:** [https://house-price-prediction-acuccze543baay6rn4354syp34bgbq.streamlit.app/](https://house-price-prediction-acuccze543baay6rn4354syp34bgbq.streamlit.app/)

---

## 🎯 Project Overview

| Item | Details |
|------|---------|
| **Task** | Regression (predict continuous house price) |
| **Dataset** | Kaggle House Prices — Ames, Iowa (1,460 records, 79 features) |
| **Target Variable** | `SalePrice` (log-transformed with `log1p` for stability) |
| **Final Model** | XGBoost Regressor |
| **Deployment** | Streamlit Cloud |

---

## 📊 Model Performance

| Model | RMSE (log scale) | R² Score |
|-------|------------------|----------|
| Linear Regression (baseline) | 0.14 | 0.86 |
| **XGBoost (final)** | **0.12** | **0.89** |

> RMSE is reported on the log-transformed target. An RMSE of 0.12 corresponds to roughly **12% average prediction error** in price, which is considered strong for the Ames dataset.

---

## 🖼️ Screenshot

![Streamlit App Screenshot](screenshot.png)

*The Streamlit interface allows users to enter property details and receive an instant price estimate.*

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** — data loading and preprocessing
- **Scikit-learn** — train/test split, scaling, metrics
- **XGBoost** — gradient boosting regression model
- **Joblib** — model serialization
- **Streamlit** — interactive web app
- **Matplotlib** — exploratory data visualization

---

## 📁 Project Structure
