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
# Forecasting Functions
#########################################

def forecast_linear_regression(series, periods):
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression()
    model.fit(x, y)
    x_future = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    forecast = model.predict(x_future)
    return forecast

def forecast_exp_smoothing(series, periods):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated")
    model_fit = model.fit()
    forecast = model_fit.forecast(periods)
    return forecast

def forecast_arima(series, periods):
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(periods)
    return forecast

def forecast_avg_change(series, periods):
    if len(series) < 2:
        return np.repeat(series.iloc[-1], periods)
    changes = series.diff().dropna()
    avg_change = changes[-6:].mean() if len(changes) >= 6 else changes.mean()
    forecast = []
    last_val = series.iloc[-1]
    for _ in range(periods):
        last_val += avg_change
        forecast.append(last_val)
    return np.array(forecast)

def forecast_input_change(series, periods, input_change):
    last_val = series.iloc[-1]
    forecast = [last_val + input_change * (i+1) for i in range(periods)]
    return np.array(forecast)

def forecast_naive(series, periods):
    return np.repeat(series.iloc[-1], periods)

# Map forecast model names to functions.
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
# Plotting Helper Function
#########################################

def plot_forecast(historical, forecast, title, yaxis_title):
    # Make sure the historical series is sorted
    historical = historical.sort_index()
    # Build forecast dates (starting the month after the last historical point)
    forecast_index = pd.date_range(start=historical.index[-1] + relativedelta(months=1),
                                   periods=len(forecast), freq='MS')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical.index, y=historical.values,
                             mode='lines+markers', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast,
                             mode='lines+markers', name='Forecast'))
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title=yaxis_title)
    st.plotly_chart(fig, use_container_width=True)

#########################################
# Data Aggregation Function
#########################################

def aggregate_data(df):
    # Pivot raw data (df columns: Month, site, specialty, AdmissionStatus, WeeksWaitCategory, PathwayCount)
    pivot = df.pivot_table(index=['specialty', 'Month'],
                           columns='WeeksWaitCategory',
                           values='PathwayCount',
                           aggfunc='sum',
                           fill_value=0).reset_index()
    # Ensure expected columns exist.
    for col in ['0-17 weeks', '18-51 weeks', '52+ weeks']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot['TotalCount'] = pivot[['0-17 weeks', '18-51 weeks', '52+ weeks']].sum(axis=1)
    pivot['PctUnder18'] = (pivot['0-17 weeks'] / pivot['TotalCount'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    pivot['PctOver52'] = (pivot['52+ weeks'] / pivot['TotalCount'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    return pivot

#########################################
# Forecast Panel for a Single Measure
#########################################

def render_forecast_panel(specialty, measure_key, historical_series, measure_label, yaxis_label, forecast_periods=15):
    st.markdown(f"### {measure_label}")
    # Layout: two columns – one narrow column for model selection, one wide for the chart.
    col1, col2 = st.columns([1,3])
    with col1:
        # The widget keys include the specialty and measure so that each one is saved separately.
        model_key = f"{specialty}_{measure_key}_model"
        if model_key not in st.session_state:
            st.session_state[model_key] = "Linear Regression"
        model_choice = st.selectbox("Forecast Model", list(forecast_models.keys()), key=model_key)
        input_val = None
        if model_choice == "Input Change":
            input_key = f"{specialty}_{measure_key}_input"
            if input_key not in st.session_state:
                st.session_state[input_key] = 0.0
            input_val = st.number_input("Input Change (absolute per month)", value=st.session_state[input_key], key=input_key)
    with col2:
        # Compute forecast using the selected model.
        forecast_values = forecast_series(historical_series, forecast_periods, model_choice, input_val)
        # Build forecast DataFrame.
        forecast_dates = pd.date_range(start=historical_series.index[-1] + relativedelta(months=1),
                                       periods=forecast_periods, freq='MS')
        forecast_df = pd.DataFrame({
            "Month": forecast_dates,
            "Forecast": forecast_values
        }).set_index("Month")
        plot_forecast(historical_series, forecast_values, f"{specialty} – {measure_label}", yaxis_label)
    
    # Show the forecast table.
    st.write("#### Forecast Numbers")
    num_cols_fc = forecast_df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(forecast_df.style.format({col: "{:.2f}" for col in num_cols_fc}))
    csv_data = forecast_df.to_csv().encode('utf-8')
    st.download_button("Download Forecast CSV", csv_data, file_name=f"{specialty}_{measure_key}_forecast.csv", mime="text/csv")
    
    # A button to save the forecast for later (so that it persists and can be aggregated on the Summary page)
    save_key = f"{specialty}_{measure_key}_save"
    if st.button("Save Forecast", key=save_key):
        if "forecasts" not in st.session_state:
            st.session_state["forecasts"] = {}
        if specialty not in st.session_state["forecasts"]:
            st.session_state["forecasts"][specialty] = {}
        st.session_state["forecasts"][specialty][measure_key] = {
            "model": model_choice,
            "input": input_val,
            "forecast_df": forecast_df
        }
        st.success("Forecast saved.")

    # Return the current forecast (so it can be used if needed)
    return forecast_df

#########################################
# Render Forecasts for One Specialty
#########################################

def render_specialty_forecasts(specialty, agg_df):
    st.header(f"Forecast for Specialty: {specialty}")
    # Extract historical data for this specialty.
    spec_data = agg_df[agg_df["specialty"] == specialty].copy()
    spec_data["Month"] = pd.to_datetime(spec_data["Month"], format="%Y-%m")
    spec_data = spec_data.sort_values("Month").set_index("Month")
    st.write("#### Historical Aggregated Data")
    num_cols = spec_data.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(spec_data.style.format({col: "{:.2f}" for col in num_cols}))
    
    # Create sub-tabs for each measure.
    tabs = st.tabs(["Total PTL Size", "0–17 Weeks Group", "% Under 18 Weeks", "% Over 52 Weeks"])
    # In this design we forecast the counts for Total and 0–17 groups,
    # and let the user also forecast percentages (even if they could be derived from counts).
    with tabs[0]:
        # Forecast TotalCount (counts)
        historical_series = spec_data["TotalCount"]
        render_forecast_panel(specialty, "TotalCount", historical_series, "Total PTL Size (Count)", "Count")
    with tabs[1]:
        # Forecast 0–17 weeks (counts)
        historical_series = spec_data["0-17 weeks"]
        render_forecast_panel(specialty, "Under18", historical_series, "0–17 Weeks Group (Count)", "Count")
    with tabs[2]:
        # Forecast % Under 18 weeks (percentage)
        historical_series = spec_data["PctUnder18"]
        render_forecast_panel(specialty, "PctUnder18", historical_series, "% Under 18 Weeks", "Percentage")
    with tabs[3]:
        # Forecast % Over 52 weeks (percentage)
        historical_series = spec_data["PctOver52"]
        render_forecast_panel(specialty, "PctOver52", historical_series, "% Over 52 Weeks", "Percentage")

#########################################
# Render Summary Forecast (Aggregating All Specialties)
#########################################

def render_summary_forecast(forecast_periods=15):
    st.header("Summary Forecast (All Specialties)")
    if "forecasts" not in st.session_state or not st.session_state["forecasts"]:
        st.warning("No specialty forecasts have been saved yet.")
        return
    all_specialties = st.session_state["forecasts"].keys()
    
    # We assume that each specialty has saved a forecast for "TotalCount", "Under18" and for the percentages.
    # For the summary of counts, we sum the forecasts for TotalCount and Under18.
    summary_df = None
    total_count_list = []
    under18_list = []
    # We also compute a weighted average for % Over 52.
    weighted_pct_over52 = None
    total_weight = 0
    summary_index = None

    for spec in all_specialties:
        spec_fc = st.session_state["forecasts"][spec]
        # We check that the needed forecasts exist.
        if ("TotalCount" in spec_fc) and ("Under18" in spec_fc) and ("PctOver52" in spec_fc):
            fc_total = spec_fc["TotalCount"]["forecast_df"]
            fc_under18 = spec_fc["Under18"]["forecast_df"]
            fc_pct_over52 = spec_fc["PctOver52"]["forecast_df"]
            # For simplicity, assume that the forecast index (Month) is the same across specialties.
            if summary_index is None:
                summary_index = fc_total.index
            # Append to lists (we assume the forecast index aligns)
            total_count_list.append(fc_total["Forecast"])
            under18_list.append(fc_under18["Forecast"])
            # For weighted average, weight by total count forecast.
            if weighted_pct_over52 is None:
                weighted_pct_over52 = fc_total["Forecast"] * fc_pct_over52["Forecast"]
            else:
                weighted_pct_over52 += fc_total["Forecast"] * fc_pct_over52["Forecast"]
            total_weight += fc_total["Forecast"]
    
    if total_count_list and under18_list:
        total_count_sum = sum(total_count_list)
        under18_sum = sum(under18_list)
        # Overall % Under 18 from aggregated counts.
        overall_pct_under18 = (under18_sum / total_count_sum) * 100
        # Overall % Over 52 by weighted average.
        overall_pct_over52 = weighted_pct_over52 / total_weight

        summary_df = pd.DataFrame({
            "TotalCount": total_count_sum,
            "Under18": under18_sum,
            "% Under 18": overall_pct_under18,
            "% Over 52": overall_pct_over52
        }, index=summary_index)
        summary_df.index.name = "ForecastMonth"
        
        st.write("#### Aggregated Forecast Numbers (by Month)")
        num_cols_fc = summary_df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(summary_df.style.format({col: "{:.2f}" for col in num_cols_fc}))
        csv_data = summary_df.to_csv().encode('utf-8')
        st.download_button("Export All Forecasts (CSV)", csv_data, file_name="summary_forecasts.csv", mime="text/csv")
        
        # Also plot overall % Under 18 and % Over 52 as separate charts.
        st.markdown("#### Overall % Under 18")
        plot_forecast(summary_df["% Under 18"], summary_df["% Under 18"], "Overall % Under 18", "Percentage")
        st.markdown("#### Overall % Over 52")
        plot_forecast(summary_df["% Over 52"], summary_df["% Over 52"], "Overall % Over 52", "Percentage")
    else:
        st.warning("Not all specialties have saved forecasts yet.")

#########################################
# Main App Layout
#########################################

st.set_page_config(layout="wide")
st.title("Hospital Waiting List Forecasting App")

# -----------------------------
# Load Data from SQL (or CSV)
# -----------------------------
# Here we assume you have loaded your SQL data into a DataFrame 'df' with columns:
# Month, site, specialty, AdmissionStatus, WeeksWaitCategory, PathwayCount
# For demonstration we create dummy data if none exists.
if 'df' not in st.session_state:
    dates = pd.date_range(start="2022-01-01", periods=24, freq='MS')
    specialties = ['Cardiology', 'Orthopedics', 'Neurology']
    data_list = []
    for spec in specialties:
        for date in dates:
            total = np.random.randint(50, 150)
            under18 = np.random.randint(0, total)
            over52 = np.random.randint(0, total - under18)
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
    #df_dummy = pd.DataFrame(data_list)
    df_dummy = pd.read_csv("PTT_PTL_23_24.csv")
    st.session_state.df = df_dummy.copy()

df = st.session_state.df.copy()
# Aggregate the raw data.
agg_df = aggregate_data(df)

# Initialize the forecast storage dictionary.
if "forecasts" not in st.session_state:
    st.session_state["forecasts"] = {}

# Build main tabs: one for each specialty plus a Summary tab.
specialties = sorted(agg_df["specialty"].unique())
tab_names = specialties + ["Summary"]
tabs = st.tabs(tab_names)

for i, name in enumerate(tab_names):
    with tabs[i]:
        if name != "Summary":
            render_specialty_forecasts(name, agg_df)
        else:
            render_summary_forecast()
