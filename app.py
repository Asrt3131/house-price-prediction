import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# ---------- تنظیمات صفحه ----------
st.set_page_config(
    page_title="پیش‌بینی قیمت خانه",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Session State ----------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "history" not in st.session_state:
    st.session_state.history = []
if "scenario" not in st.session_state:
    st.session_state.scenario = {
        "GrLivArea": 1500, "OverallQual": 6, "YearBuilt": 1990,
        "TotalBsmtSF": 800, "GarageCars": 2, "FullBath": 2
    }


# ---------- بارگذاری مدل و آمار دیتاست ----------
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/columns.pkl")
    return model, scaler, columns

@st.cache_data
def load_dataset_stats():
    """میانگین‌های دیتاست train رو حساب می‌کنه"""
    try:
        df = pd.read_csv("data/train.csv")
        stats = {
            "avg_price": df["SalePrice"].mean() if "SalePrice" in df.columns else None,
            "median_price": df["SalePrice"].median() if "SalePrice" in df.columns else None,
            "avg_gr_liv_area": df["GrLivArea"].mean(),
            "avg_overall_qual": df["OverallQual"].mean(),
            "avg_year_built": df["YearBuilt"].mean(),
            "avg_total_bsmt": df["TotalBsmtSF"].mean(),
            "avg_garage_cars": df["GarageCars"].mean(),
            "avg_full_bath": df["FullBath"].mean(),
            "price_per_sqft": (df["SalePrice"] / df["GrLivArea"]).mean() if "SalePrice" in df.columns else None,
            "total_count": len(df),
        }
        return stats
    except Exception as e:
        return None

model, scaler, columns = load_model()
dataset_stats = load_dataset_stats()


# ---------- CSS بر اساس تم ----------
def get_css(dark):
    if dark:
        bg = "#0e1117"
        card_bg = "#1e2130"
        text = "#fafafa"
        subtext = "#aaa"
        border = "#2a2d3a"
        metric_bg = "#1a1d29"
    else:
        bg = "#ffffff"
        card_bg = "#f8f9fa"
        text = "#1a1a1a"
        subtext = "#666"
        border = "#e0e0e0"
        metric_bg = "#ffffff"
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Vazirmatn', sans-serif;
            direction: rtl;
            text-align: right;
        }}
        
        .stApp {{
            background-color: {bg};
            color: {text};
        }}
        
        .main-title {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            text-align: center;
            color: {subtext};
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }}
        
        .price-card {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
            margin: 1rem 0;
        }}
        
        .price-label {{
            font-size: 1rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }}
        
        .price-value {{
            font-size: 3rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }}
        
        .price-range {{
            font-size: 0.9rem;
            opacity: 0.85;
        }}
        
        .category-badge {{
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            margin-top: 0.8rem;
        }}
        
        .metric-card {{
            background: {metric_bg};
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid {border};
        }}
        
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: {subtext};
        }}
        
        .compare-card {{
            background: {card_bg};
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid {border};
            margin: 0.5rem 0;
        }}
        
        .compare-value {{
            font-size: 1.3rem;
            font-weight: 700;
            color: {text};
        }}
        
        .compare-label {{
            font-size: 0.85rem;
            color: {subtext};
            margin-bottom: 0.3rem;
        }}
        
        .diff-positive {{
            color: #38ef7d;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .diff-negative {{
            color: #e74c3c;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .history-item {{
            background: {card_bg};
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid {border};
            margin: 0.5rem 0;
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            width: 100%;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }}
    </style>
    """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)


# ---------- توابع کمکی ----------
def predict_price(input_dict):
    full_dict = {col: 0 for col in columns}
    full_dict.update(input_dict)
    df = pd.DataFrame([full_dict])
    num_cols = [c for c in df.columns if c in scaler.feature_names_in_]
    df[num_cols] = scaler.transform(df[num_cols])
    log_price = model.predict(df)[0]
    return np.expm1(log_price)


def get_price_category(price):
    if price < 100000:
        return "اقتصادی", "#3498db", "💙"
    elif price < 200000:
        return "متوسط", "#f39c12", "💛"
    elif price < 350000:
        return "خوب", "#e67e22", "🧡"
    else:
        return "لوکس", "#e74c3c", "❤️"


def create_gauge(price):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price,
        number={'prefix': "$", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 600000], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 100000], 'color': '#d4edda'},
                {'range': [100000, 200000], 'color': '#fff3cd'},
                {'range': [200000, 350000], 'color': '#ffe5b4'},
                {'range': [350000, 600000], 'color': '#f8d7da'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': price
            }
        }
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa' if st.session_state.dark_mode else '#1a1a1a'}
    )
    return fig


def create_sensitivity_chart(base_input, base_price):
    areas = list(range(500, 4001, 250))
    prices = []
    for a in areas:
        test_input = base_input.copy()
        test_input["GrLivArea"] = a
        prices.append(predict_price(test_input))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=areas, y=prices,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        name='قیمت'
    ))
    fig.add_trace(go.Scatter(
        x=[base_input["GrLivArea"]], y=[base_price],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='انتخاب شما'
    ))
    fig.update_layout(
        title="📈 تأثیر متراژ روی قیمت",
        xaxis_title="متراژ (فوت مربع)",
        yaxis_title="قیمت (دلار)",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa' if st.session_state.dark_mode else '#1a1a1a'}
    )
    return fig


def create_comparison_chart(input_dict, user_price):
    """نمودار مقایسه با میانگین دیتاست"""
    if not dataset_stats:
        return None
    
    features = ["متراژ", "کیفیت", "سال ساخت", "زیرزمین", "پارکینگ", "حمام"]
    user_values = [
        input_dict["GrLivArea"],
        input_dict["OverallQual"],
        input_dict["YearBuilt"],
        input_dict["TotalBsmtSF"],
        input_dict["GarageCars"],
        input_dict["FullBath"],
    ]
    avg_values = [
        dataset_stats["avg_gr_liv_area"],
        dataset_stats["avg_overall_qual"],
        dataset_stats["avg_year_built"],
        dataset_stats["avg_total_bsmt"],
        dataset_stats["avg_garage_cars"],
        dataset_stats["avg_full_bath"],
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='خانه شما', x=features, y=user_values, marker_color='#667eea'),
        go.Bar(name='میانگین دیتاست', x=features, y=avg_values, marker_color='#764ba2'),
    ])
    fig.update_layout(
        barmode='group',
        title="📊 مقایسه با میانگین دیتاست",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa' if st.session_state.dark_mode else '#1a1a1a'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ تنظیمات")
    
    # دکمه تغییر تم
    theme_label = "☀️ حالت روشن" if st.session_state.dark_mode else "🌙 حالت تاریک"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    st.markdown("---")
    
    # تاریخچه
    st.markdown(f"### 💾 تاریخچه پیش‌بینی‌ها ({len(st.session_state.history)})")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"🏠 ${item['price']:,.0f} — {item['time']}"):
                st.write(f"**متراژ:** {item['input']['GrLivArea']:,} ft²")
                st.write(f"**کیفیت:** {item['input']['OverallQual']}/10")
                st.write(f"**سال ساخت:** {item['input']['YearBuilt']}")
                st.write(f"**زیرزمین:** {item['input']['TotalBsmtSF']:,} ft²")
                st.write(f"**پارکینگ:** {item['input']['GarageCars']}")
                st.write(f"**حمام:** {item['input']['FullBath']}")
                st.write(f"**دسته:** {item['category']}")
        
        col_x, col_y = st.columns(2)
        with col_x:
            if st.button("🗑️ پاک کن", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with col_y:
            history_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 دانلود",
                history_json,
                file_name=f"history_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("هنوز پیش‌بینی‌ای انجام نشده")

# ---------- هدر ----------
st.markdown('<div class="main-title">🏠 پیش‌بینی قیمت خانه</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">با هوش مصنوعی XGBoost — مدل با دقت R² = 0.89</div>', unsafe_allow_html=True)

# ---------- نمایش اطلاعات دیتاست ----------
if dataset_stats:
    st.markdown("### 📊 آمار دیتاست آموزشی")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{dataset_stats["total_count"]:,}</div><div class="metric-label">تعداد خانه</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${dataset_stats["avg_price"]:,.0f}</div><div class="metric-label">میانگین قیمت</div></div>', unsafe_allow_html=True)
    with d3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${dataset_stats["median_price"]:,.0f}</div><div class="metric-label">میانه قیمت</div></div>', unsafe_allow_html=True)
    with d4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${dataset_stats["price_per_sqft"]:,.0f}</div><div class="metric-label">میانگین قیمت/ft²</div></div>', unsafe_allow_html=True)
    st.markdown("")

# ---------- دکمه‌های سناریو ----------
st.markdown("### 🎯 سناریوهای آماده")
scenario_cols = st.columns(4)
scenarios = {
    "🏚️ اقتصادی": {"GrLivArea": 800, "OverallQual": 3, "YearBuilt": 1950, "TotalBsmtSF": 0, "GarageCars": 0, "FullBath": 1},
    "🏡 متوسط": {"GrLivArea": 1500, "OverallQual": 6, "YearBuilt": 1990, "TotalBsmtSF": 800, "GarageCars": 2, "FullBath": 2},
    "🏘️ خوب": {"GrLivArea": 2500, "OverallQual": 8, "YearBuilt": 2005, "TotalBsmtSF": 1200, "GarageCars": 3, "FullBath": 3},
    "🏰 لوکس": {"GrLivArea": 4000, "OverallQual": 10, "YearBuilt": 2015, "TotalBsmtSF": 2000, "GarageCars": 4, "FullBath": 4},
}

for i, (name, values) in enumerate(scenarios.items()):
    with scenario_cols[i]:
        if st.button(name, key=f"btn_{name}", use_container_width=True):
            st.session_state.scenario = values
            st.rerun()

# ---------- فرم ورودی ----------
st.markdown("### 📝 مشخصات خانه")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gr_liv_area = st.number_input(
            "🏠 متراژ زندگی (GrLivArea - فوت مربع)",
            min_value=300, max_value=6000,
            value=st.session_state.scenario["GrLivArea"],
            step=50,
            help="مساحت فضای زندگی بالای زمین"
        )
        overall_qual = st.slider(
            "⭐ کیفیت کلی (OverallQual)",
            min_value=1, max_value=10,
            value=st.session_state.scenario["OverallQual"],
            help="۱ = خیلی ضعیف، ۱۰ = خیلی عالی"
        )
        year_built = st.number_input(
            "📅 سال ساخت (YearBuilt)",
            min_value=1870, max_value=2025,
            value=st.session_state.scenario["YearBuilt"],
            step=1
        )
    
    with col2:
        total_bsmt = st.number_input(
            "🏗️ زیرزمین (TotalBsmtSF - فوت مربع)",
            min_value=0, max_value=3000,
            value=st.session_state.scenario["TotalBsmtSF"],
            step=50
        )
        garage_cars = st.slider(
            "🚗 ظرفیت پارکینگ (GarageCars)",
            min_value=0, max_value=4,
            value=st.session_state.scenario["GarageCars"]
        )
        full_bath = st.slider(
            "🛁 حمام کامل (FullBath)",
            min_value=0, max_value=4,
            value=st.session_state.scenario["FullBath"]
        )
    
    submitted = st.form_submit_button("🔮 پیش‌بینی قیمت", use_container_width=True)


# ---------- نمایش نتیجه ----------
if submitted:
    input_dict = {
        "GrLivArea": gr_liv_area,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "TotalBsmtSF": total_bsmt,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
    }
    
    with st.spinner("در حال تحلیل..."):
        price = predict_price(input_dict)
    
    category, color, emoji = get_price_category(price)
    
    # ذخیره در تاریخچه
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "input": input_dict.copy(),
        "price": price,
        "category": category,
    })
    
    # کارت قیمت
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">💰 قیمت تخمینی ملک شما</div>
        <div class="price-value">${price:,.0f}</div>
        <div class="price-range">محدوده تخمینی: ${price*0.9:,.0f} — ${price*1.1:,.0f}</div>
        <div class="category-badge" style="background: white; color: {color};">
            {emoji} دسته: {category}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------- مقایسه با میانگین دیتاست ----------
    if dataset_stats:
        st.markdown("### 🎯 مقایسه با میانگین دیتاست")
        
        avg_price = dataset_stats["avg_price"]
        diff = price - avg_price
        diff_pct = (diff / avg_price) * 100
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="compare-card"><div class="compare-label">خانه شما</div><div class="compare-value">${price:,.0f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="compare-card"><div class="compare-label">میانگین دیتاست</div><div class="compare-value">${avg_price:,.0f}</div></div>', unsafe_allow_html=True)
        with c3:
            diff_class = "diff-positive" if diff >= 0 else "diff-negative"
            sign = "▲" if diff >= 0 else "▼"
            st.markdown(f'<div class="compare-card"><div class="compare-label">تفاوت</div><div class="compare-value {diff_class}">{sign} {abs(diff_pct):.1f}%</div><div class="compare-label">${abs(diff):,.0f}</div></div>', unsafe_allow_html=True)
        
        # نمودار مقایسه ویژگی‌ها
        comparison_fig = create_comparison_chart(input_dict, price)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
    
    # نمودارها
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(create_gauge(price), use_container_width=True)
    with col_b:
        st.plotly_chart(create_sensitivity_chart(input_dict, price), use_container_width=True)
    
    # متریک‌ها
    st.markdown("### 📊 تحلیل ویژگی‌ها")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{gr_liv_area:,}</div><div class="metric-label">متراژ (ft²)</div></div>', unsafe_allow_html=True)
    with m2:
        price_per_sqft = price / gr_liv_area
        st.markdown(f'<div class="metric-card"><div class="metric-value">${price_per_sqft:,.0f}</div><div class="metric-label">قیمت هر ft²</div></div>', unsafe_allow_html=True)
    with m3:
        age = 2025 - year_built
        st.markdown(f'<div class="metric-card"><div class="metric-value">{age}</div><div class="metric-label">سن بنا (سال)</div></div>', unsafe_allow_html=True)
    with m4:
        total_area = gr_liv_area + total_bsmt
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_area:,}</div><div class="metric-label">مساحت کل (ft²)</div></div>', unsafe_allow_html=True)
    
    # دانلود گزارش
    st.markdown("---")
    report = f"""
گزارش پیش‌بینی قیمت خانه
تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M')}
============================================

مشخصات ملک:
- متراژ زندگی: {gr_liv_area:,} فوت مربع
- زیرزمین: {total_bsmt:,} فوت مربع
- کیفیت کلی: {overall_qual}/10
- سال ساخت: {year_built} (سن: {age} سال)
- پارکینگ: {garage_cars} خودرو
- حمام کامل: {full_bath}

نتیجه پیش‌بینی:
- قیمت تخمینی: ${price:,.0f}
- محدوده: ${price*0.9:,.0f} تا ${price*1.1:,.0f}
- دسته‌بندی: {category}
- قیمت هر فوت مربع: ${price_per_sqft:,.0f}

مقایسه با میانگین دیتاست:
- میانگین دیتاست: ${dataset_stats["avg_price"]:,.0f}
- تفاوت: {diff:+,.0f} دلار ({diff_pct:+.1f}%)
"""
    st.download_button(
        "📥 دانلود گزارش (TXT)",
        report,
        file_name=f"house_price_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )

else:
    st.info("👆 مشخصات خانه رو وارد کن و دکمه «پیش‌بینی قیمت» رو بزن")


# ---------- فوتر ----------
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.9rem;">'
    'ساخته شده با ❤️ با Streamlit و XGBoost'
    '</div>',
    unsafe_allow_html=True
)
