import streamlit as st
import pandas as pd
import sys
sys.path.append('src')

from data_preprocessing import preprocess_data, create_features
from model import SalesForecaster
import plotly.graph_objects as go

# Apple-inspired page config
st.set_page_config(
    page_title="Sales Forecasting",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Exact Apple website CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background: #f5f5f7;
        padding: 0;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Apple Navigation Bar */
    .nav-bar {
        background: rgba(0,0,0,0.8);
        backdrop-filter: saturate(180%) blur(20px);
        padding: 12px 0;
        position: sticky;
        top: 0;
        z-index: 1000;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .nav-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-logo {
        font-size: 24px;
        font-weight: 700;
        color: white;
        letter-spacing: -0.5px;
    }
    
    /* Hero Section - Apple Style */
    .hero-section {
        background: #000;
        color: white;
        text-align: center;
        padding: 80px 40px 100px 40px;
    }
    
    .hero-title {
        font-size: 64px;
        font-weight: 700;
        letter-spacing: -2px;
        margin-bottom: 12px;
        line-height: 1.05;
    }
    
    .hero-subtitle {
        font-size: 28px;
        font-weight: 400;
        color: #a1a1a6;
        margin-bottom: 20px;
    }
    
    .hero-price {
        font-size: 17px;
        color: #86868b;
        margin-top: 8px;
    }
    
    /* Section Header - Apple Style */
    .section-title {
        font-size: 48px;
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #1d1d1f;
        text-align: center;
        margin: 80px 0 40px 0;
    }
    
    .section-subtitle {
        font-size: 21px;
        color: #86868b;
        text-align: center;
        margin-bottom: 60px;
    }
    
    /* Apple Product Cards */
    .product-card {
        background: #000;
        border-radius: 28px;
        padding: 50px 40px;
        margin: 20px 0;
        color: white;
        transition: transform 0.3s ease;
        overflow: hidden;
        position: relative;
        min-height: 500px;
    }
    
    .product-card:hover {
        transform: scale(1.02);
    }
    
    .card-title {
        font-size: 40px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }
    
    .card-subtitle {
        font-size: 21px;
        color: #a1a1a6;
        margin-bottom: 12px;
    }
    
    .card-price {
        font-size: 14px;
        color: #86868b;
    }
    
    /* Metric Cards - Apple Style */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 40px 0;
    }
    
    .metric-box {
        background: #fff;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .metric-number {
        font-size: 48px;
        font-weight: 700;
        color: #1d1d1f;
        letter-spacing: -1px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #86868b;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Content Container */
    .content-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
    }
    
    /* Tabs - Apple Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #d2d2d7;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #1d1d1f;
        font-size: 17px;
        font-weight: 500;
        padding: 16px 24px;
        border-radius: 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #0071e3;
        border-bottom: 2px solid #0071e3;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: white;
        border-radius: 18px;
        padding: 30px;
        border: 2px dashed #d2d2d7;
    }
    
    /* Slider */
    .stSlider {
        padding: 20px 0;
    }
    
    /* Footer */
    .apple-footer {
        background: #f5f5f7;
        padding: 40px;
        text-align: center;
        color: #86868b;
        font-size: 12px;
        margin-top: 80px;
    }
</style>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
<div class="nav-bar">
    <div class="nav-content">
        <div class="nav-logo">Sales Forecasting</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">AI-Powered Sales Forecasting</div>
    <div class="hero-subtitle">Predict the future. Make better decisions.</div>
    <div class="hero-price">Powered by Prophet ML</div>
</div>
""", unsafe_allow_html=True)


# Main Content
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# Configuration Section
st.markdown('<div class="section-title">The latest.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Take a look at what\'s new right now.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div style="background: white; border-radius: 18px; padding: 30px; margin: 10px;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Sales Data", type=['csv'], help="CSV with date and sales columns")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div style="background: white; border-radius: 18px; padding: 30px; margin: 10px;">', unsafe_allow_html=True)
    forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('data/sales_data.csv')

df = preprocess_data(df)
df = create_features(df)

# Metrics in Apple Product Card Style
st.markdown('<div style="margin: 60px 0;">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="product-card">
        <div class="card-title">Historical Data</div>
        <div class="card-subtitle">Complete dataset.</div>
        <div class="card-price">{len(df)} records analyzed</div>
        <div style="margin-top: 60px; font-size: 80px; font-weight: 700; text-align: center;">
            {len(df)}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="product-card" style="background: linear-gradient(135deg, #0071e3 0%, #005bb5 100%);">
        <div class="card-title">Average Sales</div>
        <div class="card-subtitle">Daily performance.</div>
        <div class="card-price">Optimized for growth</div>
        <div style="margin-top: 60px; font-size: 80px; font-weight: 700; text-align: center;">
            ${df['sales'].mean():.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="product-card" style="background: linear-gradient(135deg, #bf5af2 0%, #9f4fd9 100%);">
        <div class="card-title">Peak Sales</div>
        <div class="card-subtitle">All-time high.</div>
        <div class="card-price">Maximum performance</div>
        <div style="margin-top: 60px; font-size: 80px; font-weight: 700; text-align: center;">
            ${df['sales'].max():.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# Tabs Section
st.markdown('<div style="margin-top: 80px;">', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Overview", "Forecast", "Performance"])

with tab1:
    st.markdown('<div style="margin: 40px 0;">', unsafe_allow_html=True)
    
    # Chart
    st.markdown('<div style="background: white; border-radius: 18px; padding: 40px; margin: 20px 0;">', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines',
        name='Sales',
        line=dict(color='#0071e3', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 113, 227, 0.1)'
    ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=14, color='#1d1d1f'),
        xaxis=dict(showgrid=False, title='', linecolor='#d2d2d7'),
        yaxis=dict(showgrid=True, gridcolor='#f5f5f7', title='Sales ($)', linecolor='#d2d2d7'),
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Seasonality
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div style="background: white; border-radius: 18px; padding: 30px; margin: 10px;">', unsafe_allow_html=True)
        monthly_avg = df.groupby('month')['sales'].mean().reset_index()
        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(
            x=monthly_avg['month'],
            y=monthly_avg['sales'],
            marker=dict(color='#0071e3'),
            text=monthly_avg['sales'].round(0),
            textposition='outside'
        ))
        fig_month.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#1d1d1f'),
            xaxis=dict(showgrid=False, title='Month'),
            yaxis=dict(showgrid=True, gridcolor='#f5f5f7', title='Avg Sales'),
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_month, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="background: white; border-radius: 18px; padding: 30px; margin: 10px;">', unsafe_allow_html=True)
        weekly_avg = df.groupby('dayofweek')['sales'].mean().reset_index()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_avg['day_name'] = weekly_avg['dayofweek'].apply(lambda x: days[x])
        
        fig_week = go.Figure()
        fig_week.add_trace(go.Bar(
            x=weekly_avg['day_name'],
            y=weekly_avg['sales'],
            marker=dict(color='#bf5af2'),
            text=weekly_avg['sales'].round(0),
            textposition='outside'
        ))
        fig_week.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#1d1d1f'),
            xaxis=dict(showgrid=False, title='Day'),
            yaxis=dict(showgrid=True, gridcolor='#f5f5f7', title='Avg Sales'),
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_week, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div style="margin: 40px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="background: white; border-radius: 18px; padding: 40px;">', unsafe_allow_html=True)
    
    with st.spinner("Training AI model..."):
        forecaster = SalesForecaster()
        model = forecaster.train_prophet(df, date_column='date', sales_column='sales')
        forecast = forecaster.predict(periods=forecast_days, freq='D')
    
    st.success("✓ Forecast complete")
    
    # Forecast chart
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines',
        name='Historical',
        line=dict(color='#0071e3', width=2)
    ))
    
    future_forecast = forecast[forecast['ds'] > df['date'].max()]
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#bf5af2', width=3)
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(191, 90, 242, 0.2)',
        fill='tonexty',
        name='Confidence',
        hoverinfo='skip'
    ))
    
    fig_forecast.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=14, color='#1d1d1f'),
        xaxis=dict(showgrid=False, title='', linecolor='#d2d2d7'),
        yaxis=dict(showgrid=True, gridcolor='#f5f5f7', title='Sales ($)', linecolor='#d2d2d7'),
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div style="margin: 40px 0;">', unsafe_allow_html=True)
    
    historical_forecast = forecast[forecast['ds'].isin(df['date'])]
    actual_values = df[df['date'].isin(historical_forecast['ds'])]['sales'].values
    predicted_values = historical_forecast['yhat'].values
    
    metrics = forecaster.evaluate(actual_values, predicted_values)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{metrics['MAE']:.0f}</div>
            <div class="metric-label">MAE</div>
            <p style="font-size: 12px; color: #86868b; margin-top: 8px;">Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{metrics['RMSE']:.0f}</div>
            <div class="metric-label">RMSE</div>
            <p style="font-size: 12px; color: #86868b; margin-top: 8px;">Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number">{metrics['MAPE']:.1f}%</div>
            <div class="metric-label">MAPE</div>
            <p style="font-size: 12px; color: #86868b; margin-top: 8px;">Mean Absolute % Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="apple-footer">
    <p>Built with Prophet & Streamlit</p>
    <p style="margin-top: 8px;">Future Interns • Machine Learning Track</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
