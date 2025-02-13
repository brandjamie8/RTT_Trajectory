import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

#########################################
# Forecasting functions
#########################################

def forecast_linear_regression(series, periods):
    # Convert index to a numeric time index
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression()
    model.fit(x, y)
    x_future = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    forecast = model.predict(x_future)
    return forecast

def forecast_exp_smoothing(series, periods):
    # Here we assume additive trend and seasonality (seasonal_periods=12)
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated")
    model_fit = model.fit()
    forecast = model_fit.forecast(periods)
    return forecast

def forecast_arima(series, periods):
    # Using an ARIMA(1,1,1) model as an example; you can adjust the order or use auto_arima.
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(periods)
    return forecast

def forecast_avg_change(series, periods):
    # Compute the average change over the last 6 months (or all available if fewer)
    if len(series) < 2:
        return np.repeat(series.iloc[-1], periods)
    changes = series.diff().dropna()
    avg_change = changes[-6:].mean() if len(changes) >= 6 else changes.mean()
    forecast = []
    last_val = series.iloc[-1]
    for i in range(periods):
        last_val = last_val + avg_change
        forecast.append(last_val)
    return np.array(forecast)

def forecast_input_change(series, periods, input_change):
    # Forecast using an input absolute change per month
    last_val = series.iloc[-1]
    forecast = [last_val + input_change * (i + 1) for i in range(periods)]
    return np.array(forecast)

def forecast_naive(series, periods):
    # Simply extend the last observed value into the future
    return np.repeat(series.iloc[-1], periods)

# Map model names to functions
forecast_models = {
    "Linear Regression": forecast_linear_regression,
    "Exponential Smoothing": forecast_exp_smoothing,
    "ARIMA": forecast_arima,
    "Average Change (last 6 months)": forecast_avg_change,
    "Input Change": forecast_input_change,
    "Naive (Last Value)": forecast_naive
}

def forecast_series(series, periods, model_name, input_change=None):
    if model_name == "Input Change":
        return forecast_models[model_name](series, periods, input_change)
    else:
        return forecast_models[model_name](series, periods)

#########################################
# Helper plotting function
#########################################

def plot_forecast(historical, forecast, title, yaxis_title):
    # Ensure the historical series is sorted by date
    historical = historical.sort_index()
    # Create forecast dates: starting the month after the last historical point
    forecast_index = pd.date_range(start=historical.index[-1] + relativedelta(months=1),
                                   periods=len(forecast), freq='MS')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical.index, y=historical.values,
                             mode='lines+markers', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast,
                             mode='lines+markers', name='Forecast'))
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title=yaxis_title)
    st.plotly_chart(fig)

#########################################
# Data aggregation
#########################################

# This function aggregates your raw data (with columns:
# Month, site, specialty, AdmissionStatus, WeeksWaitCategory, PathwayCount)
# into one row per specialty per month with the following measures:
#  • TotalCount
#  • Count for '0-17 weeks'
#  • Count for '52+ weeks'
#  • % Under 18 weeks and % Over 52 weeks.
def aggregate_data(df):
    pivot = df.pivot_table(index=['specialty', 'Month'],
                           columns='WeeksWaitCategory',
                           values='PathwayCount',
                           aggfunc='sum',
                           fill_value=0).reset_index()
    # Ensure the expected columns exist:
    for col in ['0-17 weeks', '18-51 weeks', '52+ weeks']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot['TotalCount'] = pivot[['0-17 weeks', '18-51 weeks', '52+ weeks']].sum(axis=1)
    pivot['PctUnder18'] = (pivot['0-17 weeks'] / pivot['TotalCount'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    pivot['PctOver52'] = (pivot['52+ weeks'] / pivot['TotalCount'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    return pivot

#########################################
# Streamlit App Layout
#########################################

st.title("Hospital Waiting List Forecasting App")

# -----------------------------
# Load your data here
# -----------------------------
# For example, you might load the data from SQL into a DataFrame.
# In this example we assume the DataFrame 'df' has these columns:
# Month, site, specialty, AdmissionStatus, WeeksWaitCategory, PathwayCount
#
# If you have a CSV from SQL export, you might do:
# df = pd.read_csv('your_data.csv')
#
# For demonstration purposes, we generate dummy aggregated data.
if 'df' not in st.session_state:
    # Create dummy data for 24 months and 3 specialties
    dates = pd.date_range(start="2022-01-01", periods=24, freq='MS')
    specialties = ['Cardiology', 'Orthopedics', 'Neurology']
    data_list = []
    for spec in specialties:
        for date in dates:
            total = np.random.randint(50, 150)
            under18 = np.random.randint(0, total)
            over52 = np.random.randint(0, total - under18)
            # The remaining goes to 18-51 weeks
            data_list.append({
                "specialty": spec,
                "Month": date.strftime('%Y-%m'),
                "WeeksWaitCategory": "0-17 weeks",
                "PathwayCount": under18
            })
            data_list.append({
                "specialty": spec,
                "Month": date.strftime('%Y-%m'),
                "WeeksWaitCategory": "18-51 weeks",
                "PathwayCount": total - under18 - over52
            })
            data_list.append({
                "specialty": spec,
                "Month": date.strftime('%Y-%m'),
                "WeeksWaitCategory": "52+ weeks",
                "PathwayCount": over52
            })
    df_dummy = pd.DataFrame(data_list)
    st.session_state.df = df_dummy.copy()

# Use the DataFrame from session state
df = st.session_state.df.copy()
# Convert Month to datetime (assumes format YYYY-MM)
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

# Create an aggregated/pivoted DataFrame
agg_df = aggregate_data(df)
# Convert Month to datetime if not already
agg_df['Month'] = pd.to_datetime(agg_df['Month'], format='%Y-%m')

#########################################
# Sidebar Navigation: Summary or Specialty Page
#########################################

page = st.sidebar.selectbox("Select Page", ["Summary", "Specialty"])

if page == "Specialty":
    # Let the user select a specialty
    specialties = sorted(agg_df['specialty'].unique())
    selected_specialty = st.sidebar.selectbox("Select Specialty", specialties)
    spec_data = agg_df[agg_df['specialty'] == selected_specialty].copy()
    spec_data = spec_data.sort_values("Month").set_index("Month")
    
    st.header(f"Specialty: {selected_specialty}")
    st.subheader("Historical Aggregated Data")
    st.dataframe(spec_data)
    
    # Select forecast model (applied to all four charts)
    model_choice = st.sidebar.selectbox("Select Forecast Model", list(forecast_models.keys()))
    input_change_value = None
    if model_choice == "Input Change":
        input_change_value = st.sidebar.number_input("Input Change (absolute per month)", value=0.0)
    
    forecast_periods = 15  # next 15 months
    
    # --- Total PTL Size Chart ---
    total_series = spec_data['TotalCount']
    forecast_total = forecast_series(total_series, forecast_periods, model_choice, input_change_value)
    st.subheader("Total PTL Size Forecast")
    plot_forecast(total_series, forecast_total, f"Total PTL Size for {selected_specialty}", "Count")
    
    # --- 0–17 Weeks Group Chart ---
    under18_series = spec_data['0-17 weeks']
    forecast_under18 = forecast_series(under18_series, forecast_periods, model_choice, input_change_value)
    st.subheader("0–17 Weeks Group Forecast")
    plot_forecast(under18_series, forecast_under18, f"0–17 Weeks Group for {selected_specialty}", "Count")
    
    # --- % Under 18 Weeks Chart ---
    pct_under18_series = spec_data['PctUnder18']
    forecast_pct_under18 = forecast_series(pct_under18_series, forecast_periods, model_choice, input_change_value)
    st.subheader("% Under 18 Weeks Forecast")
    plot_forecast(pct_under18_series, forecast_pct_under18, f"% Under 18 Weeks for {selected_specialty}", "Percentage")
    
    # --- % Over 52 Weeks Chart ---
    pct_over52_series = spec_data['PctOver52']
    forecast_pct_over52 = forecast_series(pct_over52_series, forecast_periods, model_choice, input_change_value)
    st.subheader("% Over 52 Weeks Forecast")
    plot_forecast(pct_over52_series, forecast_pct_over52, f"% Over 52 Weeks for {selected_specialty}", "Percentage")
    
elif page == "Summary":
    st.header("Summary for All Specialties")
    # Aggregate across all specialties by Month
    summary_df = agg_df.groupby("Month").agg({
        "TotalCount": "sum",
        "0-17 weeks": "sum",
        "52+ weeks": "sum"
    }).reset_index()
    summary_df['PctUnder18'] = summary_df['0-17 weeks'] / summary_df['TotalCount'] * 100
    summary_df['PctOver52'] = summary_df['52+ weeks'] / summary_df['TotalCount'] * 100
    summary_df = summary_df.sort_values("Month").set_index("Month")
    
    st.subheader("Historical Summary Data")
    st.dataframe(summary_df)
    
    model_choice = st.sidebar.selectbox("Select Forecast Model for Summary", list(forecast_models.keys()))
    input_change_value = None
    if model_choice == "Input Change":
        input_change_value = st.sidebar.number_input("Input Change (absolute per month) for Summary", value=0.0)
    forecast_periods = 15
    
    # --- Overall Total PTL Size Forecast ---
    total_series = summary_df['TotalCount']
    forecast_total = forecast_series(total_series, forecast_periods, model_choice, input_change_value)
    st.subheader("Overall Total PTL Size Forecast")
    plot_forecast(total_series, forecast_total, "Overall Total PTL Size", "Count")
    
    # --- Overall % Under 18 Weeks Forecast ---
    pct_under18_series = summary_df['PctUnder18']
    forecast_pct_under18 = forecast_series(pct_under18_series, forecast_periods, model_choice, input_change_value)
    st.subheader("Overall % Under 18 Weeks Forecast")
    plot_forecast(pct_under18_series, forecast_pct_under18, "Overall % Under 18 Weeks", "Percentage")
    
    # --- Overall % Over 52 Weeks Forecast ---
    pct_over52_series = summary_df['PctOver52']
    forecast_pct_over52 = forecast_series(pct_over52_series, forecast_periods, model_choice, input_change_value)
    st.subheader("Overall % Over 52 Weeks Forecast")
    plot_forecast(pct_over52_series, forecast_pct_over52, "Overall % Over 52 Weeks", "Percentage")
