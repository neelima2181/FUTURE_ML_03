import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

st.set_page_config(page_title="Sales Forecasting System", layout="wide", page_icon="📈")

# Custom CSS injection for KPI cards and layout
st.markdown("""
<style>
    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #1E222D;
        border: 1px solid #2D3748;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="metric-container"] label {
        color: #00F0FF !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    /* Main block padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Sales Demand Forecasting System")
st.markdown("<p style='font-size: 15px; color: #A0AEC0;'>This platform leverages machine learning to track long-term business performance and forecast proactive retail demand, helping you optimize inventory and staffing.</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists('sales_data.csv'):
        df = pd.read_csv('sales_data.csv', parse_dates=['Date'])
        return df
    elif os.path.exists('superstore.csv'):
        df = pd.read_csv('superstore.csv', parse_dates=['Date'])
        return df
    else:
        st.error("Data file not found. Please ensure 'sales_data.csv' or 'superstore.csv' exists in the directory.")
        return None

df = load_data()

if df is not None:
    # Feature Engineering
    df_feat = df.copy()
    df_feat['Year'] = df_feat['Date'].dt.year
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Day'] = df_feat['Date'].dt.day
    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek
    df_feat['IsWeekend'] = df_feat['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df_feat['Sales_Lag1'] = df_feat['Sales'].shift(1)
    df_feat['Sales_Lag7'] = df_feat['Sales'].shift(7)
    df_feat = df_feat.dropna()
    
    # Target and Features
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Sales_Lag1', 'Sales_Lag7']
    X = df_feat[features]
    y = df_feat['Sales']
    
    # Sidebar constraints
    st.sidebar.header("Forecast Settings")
    test_size = st.sidebar.slider("Forecast Horizon (Days)", min_value=14, max_value=120, value=60, step=1)
    
    # Train-Test Split based on horizon
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    dates_train = df_feat['Date'][:-test_size]
    dates_test = df_feat['Date'][-test_size:]
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate New Metrics
    # 1. Average Monthly Sales
    monthly_sales_history = df_feat.iloc[:-test_size].groupby(['Year', 'Month'])['Sales'].sum()
    avg_monthly_sales = int(monthly_sales_history.mean())
    format_avg_monthly_sales = f"${avg_monthly_sales:,.0f}"
    
    # 2. Annual Trend Growth
    if len(y_train) >= 730:
        sales_last_year = y_train[-365:].mean()
        sales_prev_year = y_train[-730:-365].mean()
        annual_growth = ((sales_last_year - sales_prev_year) / sales_prev_year) * 100
        growth_str = f"{annual_growth:+.1f}%"
    else:
        growth_str = "N/A"
        
    # 3. Next Month Forecast
    next_month_total = int(y_pred[:30].sum()) if len(y_pred) >= 30 else int(y_pred.sum())
    format_next_month_total = f"${next_month_total:,.0f}"
    
    # 4. Peak Season
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    peak_month_num = df_feat.groupby('Month')['Sales'].mean().idxmax()
    peak_season = month_names[peak_month_num]

    # Top Row Metrics Layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Average Monthly Sales", value=format_avg_monthly_sales)
    with col2:
        st.metric(label="Annual Trend Growth", value=growth_str)
    with col3:
        st.metric(label="Next Month Forecast", value=format_next_month_total)
    with col4:
        st.metric(label="Peak Season", value=peak_season)
        
    st.markdown("---")
    
    # Interactive Plot
    
    # Create DataFrames for resampling to Monthly aggregates
    train_df = pd.DataFrame({'Date': dates_train, 'Sales': y_train}).set_index('Date')
    test_df = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Forecast': y_pred}).set_index('Date')
    
    train_monthly = train_df.resample('ME').sum()
    test_monthly  = test_df.resample('ME').sum()
    
    history_months = st.slider("Historical Data to Display (Months)", min_value=12, max_value=60, value=36)
    plot_train_monthly = train_monthly.iloc[-history_months:].copy()
    
    # Prepare data for Altair
    plot_df = plot_train_monthly.rename(columns={'Sales': 'Historical Sales'})
    if not test_monthly.empty:
        test_df_plot = test_monthly.rename(columns={'Actual': 'Actual Future Sales', 'Forecast': 'Forecasted Sales'})
        combined = plot_df.join(test_df_plot, how='outer')
    else:
        combined = plot_df
    
    melted = combined.reset_index().melt(id_vars='Date', var_name='Type', value_name='Sales').dropna()
    
    chart_line = alt.Chart(melted).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('Date:T', title='Month & Year', axis=alt.Axis(format='%b %Y', labelAngle=-45, titleColor='#A0AEC0', labelColor='#A0AEC0')),
        y=alt.Y('Sales:Q', title='Total Sales ($)', axis=alt.Axis(titleColor='#A0AEC0', labelColor='#A0AEC0')),
        color=alt.Color('Type:N', scale=alt.Scale(
            domain=['Historical Sales', 'Actual Future Sales', 'Forecasted Sales'],
            range=['#A0AEC0', '#00F0FF', '#FF9800']
        ), legend=alt.Legend(title="", orient='top', titleFontWeight='bold', labelColor='#FFFFFF')),
        strokeDash=alt.condition(
            alt.datum.Type == 'Forecasted Sales',
            alt.value([5, 5]),
            alt.value([0])
        ),
        tooltip=[alt.Tooltip('Date:T', format='%Y-%m', title='Date'), 'Type:N', alt.Tooltip('Sales:Q', format='$,.0f', title='Sales')]
    ).interactive().properties(height=450)
    
    st.altair_chart(chart_line, width="stretch")
    
    st.markdown("---")
    
    # Seasonal Analytics
    st.subheader("Seasonal Analytics")
    
    st.write("**Average Seasonal Strength by Month**")
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    sales_by_month = df_feat.groupby('Month')['Sales'].mean().reset_index()
    sales_by_month['Month Name'] = sales_by_month['Month'].map(month_names)
    
    chart_bar = alt.Chart(sales_by_month).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Month Name:N', sort=list(month_names.values()), title='Month', axis=alt.Axis(labelAngle=0, titleColor='#A0AEC0', labelColor='#A0AEC0')),
        y=alt.Y('Sales:Q', title='Avg Monthly Strength ($)', axis=alt.Axis(titleColor='#A0AEC0', labelColor='#A0AEC0')),
        color=alt.Color('Sales:Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['Month Name:N', alt.Tooltip('Sales:Q', format='$,.0f', title='Avg Strength')]
    ).properties(height=350)
    
    st.altair_chart(chart_bar, width="stretch")

    st.markdown("---")
    
    # Raw Data Expander
    with st.expander("View Raw Data"):
        st.dataframe(df.sort_values('Date', ascending=False).head(200))
