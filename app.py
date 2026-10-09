import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# ---------- Page Config ----------
st.set_page_config(
    page_title="House Price Prediction",
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


# ---------- Load Model & Dataset Stats ----------
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/columns.pkl")
    return model, scaler, columns

@st.cache_data
def load_dataset_stats():
    """Compute averages from the training dataset."""
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


# ---------- CSS based on theme ----------
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
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


# ---------- Helper Functions ----------
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
        return "Budget", "#3498db", "💙"
    elif price < 200000:
        return "Average", "#f39c12", "💛"
    elif price < 350000:
        return "Good", "#e67e22", "🧡"
    else:
        return "Luxury", "#e74c3c", "❤️"


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
        name='Predicted Price'
    ))
    fig.add_trace(go.Scatter(
        x=[base_input["GrLivArea"]], y=[base_price],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Your Selection'
    ))
    fig.update_layout(
        title="📈 Price Sensitivity to Living Area",
        xaxis_title="Living Area (sq ft)",
        yaxis_title="Price (USD)",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa' if st.session_state.dark_mode else '#1a1a1a'}
    )
    return fig


def create_comparison_chart(input_dict, user_price):
    """Compare user's house features with dataset averages."""
    if not dataset_stats:
        return None
    
    features = ["Area", "Quality", "Year Built", "Basement", "Garage", "Bath"]
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
        go.Bar(name='Your House', x=features, y=user_values, marker_color='#667eea'),
        go.Bar(name='Dataset Average', x=features, y=avg_values, marker_color='#764ba2'),
    ])
    fig.update_layout(
        barmode='group',
        title="📊 Comparison with Dataset Averages",
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
    st.markdown("### ⚙️ Settings")
    
    # Theme toggle
    theme_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    st.markdown("---")
    
    # History
    st.markdown(f"### 💾 Prediction History ({len(st.session_state.history)})")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"🏠 ${item['price']:,.0f} — {item['time']}"):
                st.write(f"**Area:** {item['input']['GrLivArea']:,} sq ft")
                st.write(f"**Quality:** {item['input']['OverallQual']}/10")
                st.write(f"**Year Built:** {item['input']['YearBuilt']}")
                st.write(f"**Basement:** {item['input']['TotalBsmtSF']:,} sq ft")
                st.write(f"**Garage:** {item['input']['GarageCars']}")
                st.write(f"**Bath:** {item['input']['FullBath']}")
                st.write(f"**Category:** {item['category']}")
        
        col_x, col_y = st.columns(2)
        with col_x:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with col_y:
            history_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 Download",
                history_json,
                file_name=f"history_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("No predictions yet.")

# ---------- Header ----------
st.markdown('<div class="main-title">🏠 House Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by XGBoost — Model Accuracy R² = 0.89</div>', unsafe_allow_html=True)

# ---------- Dataset Statistics ----------
if dataset_stats:
    st.markdown("### 📊 Training Dataset Statistics")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{dataset_stats["total_count"]:,}</div><div class="metric-label">Total Houses</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${dataset_stats["avg_price"]:,.0f}</div><div class="metric-label">Average Price</div></div>', unsafe_allow_html=True)
    with d3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${dataset_stats["median_price"]:,.0f}</div><div class="metric-label">Median Price</div></div>', unsafe_allow_html=True)
    with d4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${dataset_stats["price_per_sqft"]:,.0f}</div><div class="metric-label">Avg Price / ft²</div></div>', unsafe_allow_html=True)
    st.markdown("")

# ---------- Quick Scenarios ----------
st.markdown("### 🎯 Quick Scenarios")
scenario_cols = st.columns(4)
scenarios = {
    "🏚️ Budget": {"GrLivArea": 800, "OverallQual": 3, "YearBuilt": 1950, "TotalBsmtSF": 0, "GarageCars": 0, "FullBath": 1},
    "🏡 Average": {"GrLivArea": 1500, "OverallQual": 6, "YearBuilt": 1990, "TotalBsmtSF": 800, "GarageCars": 2, "FullBath": 2},
    "🏘️ Good": {"GrLivArea": 2500, "OverallQual": 8, "YearBuilt": 2005, "TotalBsmtSF": 1200, "GarageCars": 3, "FullBath": 3},
    "🏰 Luxury": {"GrLivArea": 4000, "OverallQual": 10, "YearBuilt": 2015, "TotalBsmtSF": 2000, "GarageCars": 4, "FullBath": 4},
}

for i, (name, values) in enumerate(scenarios.items()):
    with scenario_cols[i]:
        if st.button(name, key=f"btn_{name}", use_container_width=True):
            st.session_state.scenario = values
            st.rerun()

# ---------- Input Form ----------
st.markdown("### 📝 House Details")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gr_liv_area = st.number_input(
            "🏠 Living Area (GrLivArea - sq ft)",
            min_value=300, max_value=6000,
            value=st.session_state.scenario["GrLivArea"],
            step=50,
            help="Above-ground living area in square feet"
        )
        overall_qual = st.slider(
            "⭐ Overall Quality (OverallQual)",
            min_value=1, max_value=10,
            value=st.session_state.scenario["OverallQual"],
            help="1 = Very Poor, 10 = Very Excellent"
        )
        year_built = st.number_input(
            "📅 Year Built (YearBuilt)",
            min_value=1870, max_value=2025,
            value=st.session_state.scenario["YearBuilt"],
            step=1
        )
    
    with col2:
        total_bsmt = st.number_input(
            "🏗️ Basement Area (TotalBsmtSF - sq ft)",
            min_value=0, max_value=3000,
            value=st.session_state.scenario["TotalBsmtSF"],
            step=50
        )
        garage_cars = st.slider(
            "🚗 Garage Capacity (GarageCars)",
            min_value=0, max_value=4,
            value=st.session_state.scenario["GarageCars"]
        )
        full_bath = st.slider(
            "🛁 Full Bathrooms (FullBath)",
            min_value=0, max_value=4,
            value=st.session_state.scenario["FullBath"]
        )
    
    submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)


# ---------- Show Result ----------
if submitted:
    input_dict = {
        "GrLivArea": gr_liv_area,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "TotalBsmtSF": total_bsmt,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
    }
    
    with st.spinner("Analyzing..."):
        price = predict_price(input_dict)
    
    category, color, emoji = get_price_category(price)
    
    # Save to history
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "input": input_dict.copy(),
        "price": price,
        "category": category,
    })
    
    # Price card
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">💰 Estimated Property Price</div>
        <div class="price-value">${price:,.0f}</div>
        <div class="price-range">Estimated Range: ${price*0.9:,.0f} — ${price*1.1:,.0f}</div>
        <div class="category-badge" style="background: white; color: {color};">
            {emoji} Category: {category}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------- Comparison with Dataset ----------
    if dataset_stats:
        st.markdown("### 🎯 Comparison with Dataset Averages")
        
        avg_price = dataset_stats["avg_price"]
        diff = price - avg_price
        diff_pct = (diff / avg_price) * 100
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="compare-card"><div class="compare-label">Your House</div><div class="compare-value">${price:,.0f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="compare-card"><div class="compare-label">Dataset Average</div><div class="compare-value">${avg_price:,.0f}</div></div>', unsafe_allow_html=True)
        with c3:
            diff_class = "diff-positive" if diff >= 0 else "diff-negative"
            sign = "▲" if diff >= 0 else "▼"
            st.markdown(f'<div class="compare-card"><div class="compare-label">Difference</div><div class="compare-value {diff_class}">{sign} {abs(diff_pct):.1f}%</div><div class="compare-label">${abs(diff):,.0f}</div></div>', unsafe_allow_html=True)
        
        # Comparison bar chart
        comparison_fig = create_comparison_chart(input_dict, price)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(create_gauge(price), use_container_width=True)
    with col_b:
        st.plotly_chart(create_sensitivity_chart(input_dict, price), use_container_width=True)
    
    # Metrics
    st.markdown("### 📊 Feature Analysis")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{gr_liv_area:,}</div><div class="metric-label">Living Area (ft²)</div></div>', unsafe_allow_html=True)
    with m2:
        price_per_sqft = price / gr_liv_area
        st.markdown(f'<div class="metric-card"><div class="metric-value">${price_per_sqft:,.0f}</div><div class="metric-label">Price per ft²</div></div>', unsafe_allow_html=True)
    with m3:
        age = 2025 - year_built
        st.markdown(f'<div class="metric-card"><div class="metric-value">{age}</div><div class="metric-label">Building Age (years)</div></div>', unsafe_allow_html=True)
    with m4:
        total_area = gr_liv_area + total_bsmt
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_area:,}</div><div class="metric-label">Total Area (ft²)</div></div>', unsafe_allow_html=True)
    
    # Report download
    st.markdown("---")
    report = f"""
House Price Prediction Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
============================================

Property Details:
- Living Area: {gr_liv_area:,} sq ft
- Basement: {total_bsmt:,} sq ft
- Overall Quality: {overall_qual}/10
- Year Built: {year_built} (Age: {age} years)
- Garage: {garage_cars} cars
- Full Bathrooms: {full_bath}

Prediction Result:
- Estimated Price: ${price:,.0f}
- Range: ${price*0.9:,.0f} to ${price*1.1:,.0f}
- Category: {category}
- Price per sq ft: ${price_per_sqft:,.0f}

Comparison with Dataset:
- Dataset Average: ${dataset_stats["avg_price"]:,.0f}
- Difference: {diff:+,.0f} USD ({diff_pct:+.1f}%)
"""
    st.download_button(
        "📥 Download Report (TXT)",
        report,
        file_name=f"house_price_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )

else:
    st.info("👆 Enter the house details and click 'Predict Price'")


# ---------- Footer ----------
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.9rem;">'
    'Built with ❤️ using Streamlit and XGBoost'
    '</div>',
    unsafe_allow_html=True
)


# ==================== Footer ====================
st.markdown("---")
st.markdown("""
    <div class="footer">
        Built by <span class="name">A_srt343</span><br>
        Instagram: <span class="name">@project_srt343</span>
    </div>
""", unsafe_allow_html=True)
