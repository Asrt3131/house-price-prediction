# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house prices based on property features such as living area, overall quality, year built, basement size, garage capacity, and number of bathrooms.

The model is trained on the **Kaggle House Prices (Ames) dataset** and deployed as an interactive **Streamlit** web app with a modern UI, dark/light theme, dataset comparison, and prediction history.

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

## ✨ Features

### 🤖 Machine Learning
- **XGBoost Regressor** trained on log-transformed `SalePrice`
- **Log1p transformation** for stable regression on skewed prices
- **Feature scaling** via `StandardScaler` for numeric columns
- **Categorical encoding** with `pd.get_dummies`
- Automatic inverse transform (`expm1`) to return real prices

### 🎨 Interactive Web App (Streamlit)
- **Modern RTL UI** with Persian (Farsi) support and `Vazirmatn` font
- **Dark / Light theme toggle** — instantly switch themes from the sidebar
- **Custom gradient design** — animated cards, badges, and metrics
- **4 Quick Scenarios** — fill the form with one click (Budget, Average, Good, Luxury)
- **Real-time prediction** with smooth spinner feedback
- **Responsive layout** — works on desktop and mobile

### 📊 Visual Analytics
- **Gauge Chart** — shows predicted price on a colored scale (economic → luxury)
- **Sensitivity Chart** — shows how price changes as living area varies
- **Comparison Bar Chart** — compares user's house features vs. dataset averages
- **Metric Cards** — area, price per ft², building age, total area

### 🎯 Dataset Comparison
- Displays dataset-wide statistics: **count, average price, median price, average price/ft²**
- Compares user's input directly to dataset averages
- Shows **percentage difference** (▲ or ▼) from the average price
- Feature-by-feature bar chart (area, quality, year, basement, garage, bath)

### 💾 Prediction History
- Automatically saves every prediction in **session state**
- Shows last 10 predictions in the **sidebar** with expandable details
- **Clear history** button
- **Download history as JSON** for offline use

### 📥 Reporting
- **Download full report** as a `.txt` file with all input features and prediction details
- Includes comparison with dataset averages

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

*The Streamlit interface allows users to enter property details and receive an instant price estimate, with visual analytics and dataset comparison.*

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** — data loading and preprocessing
- **Scikit-learn** — train/test split, scaling, metrics
- **XGBoost** — gradient boosting regression model
- **Joblib** — model serialization
- **Streamlit** — interactive web app
- **Plotly** — interactive charts (gauge, sensitivity, comparison)
- **Matplotlib** — exploratory data visualization

---

## 📁 Project Structure
