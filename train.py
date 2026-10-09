import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ۱. خواندن دیتا
df = pd.read_csv("data/train.csv")
target = "SalePrice"

# ۲. حذف ستون Id (بی‌استفاده)
df = df.drop(columns=["Id"])

# ۳. جدا کردن ویژگی‌های عددی و دسته‌ای
num_cols = df.select_dtypes(include=[np.number]).columns.drop(target)
cat_cols = df.select_dtypes(include=["object"]).columns

# ۴. پر کردن مقادیر گم‌شده
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna("None")

# ۵. انکود کردن متغیرهای دسته‌ای (One-Hot Encoding)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ۶. جدا کردن X و y
X = df.drop(columns=[target])
y = np.log1p(df[target])  # لگاریتم برای نرمال‌سازی قیمت

# ۷. اسکیل کردن ویژگی‌های عددی
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ۸. ذخیره لیست ستون‌ها و اسکیلر برای Streamlit
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/columns.pkl")

# ۹. تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ۱۰. آموزش مدل
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# ۱۱. ارزیابی
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"RMSE (log scale): {rmse:.4f}")
print(f"R²: {r2:.4f}")

# ۱۲. ذخیره مدل
joblib.dump(model, "models/model.pkl")
print("مدل و پیش‌پردازشگرها ذخیره شدند.")