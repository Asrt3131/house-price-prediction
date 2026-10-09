import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_model()

st.title("🏠 پیش‌بینی قیمت خانه")
st.write("مشخصات خانه را وارد کنید تا قیمت تخمینی را ببینید.")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("متراژ زندگی (GrLivArea)", 500, 6000, 1500)
    overall_qual = st.slider("کیفیت کلی (OverallQual)", 1, 10, 5)
    year_built = st.number_input("سال ساخت (YearBuilt)", 1870, 2020, 1990)

with col2:
    total_bsmt = st.number_input("زیرزمین (TotalBsmtSF)", 0, 3000, 800)
    garage_cars = st.slider("ظرفیت پارکینگ (GarageCars)", 0, 4, 2)
    full_bath = st.slider("حمام (FullBath)", 0, 3, 1)

if st.button("پیش‌بینی قیمت"):
    # ساخت یک ردیف با تمام ستون‌های مدل
    input_dict = {col: 0 for col in columns}
    
    # پر کردن مقادیر عددی شناخته‌شده
    input_dict["GrLivArea"] = gr_liv_area
    input_dict["OverallQual"] = overall_qual
    input_dict["YearBuilt"] = year_built
    input_dict["TotalBsmtSF"] = total_bsmt
    input_dict["GarageCars"] = garage_cars
    input_dict["FullBath"] = full_bath
    
    input_df = pd.DataFrame([input_dict])
    
    # اسکیل کردن ستون‌های عددی
    num_cols_to_scale = [c for c in input_df.columns if c in scaler.feature_names_in_]
    input_df[num_cols_to_scale] = scaler.transform(input_df[num_cols_to_scale])
    
    # پیش‌بینی (برگرداندن از لگاریتم)
    log_price = model.predict(input_df)[0]
    price = np.expm1(log_price)
    
    st.success(f"💰 قیمت تخمینی: **${price:,.0f}**")