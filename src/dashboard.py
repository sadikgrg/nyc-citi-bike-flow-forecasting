import geopandas as gpd
import pandas as pd
import streamlit as st
from sqlalchemy import text
from dashboard_charts import create_map, create_demand_trend, create_imbalance_ranking
from database import get_engine


FORECAST_TABLE = "hourly_neighborhood_forecast"
WEATHER_TABLE = "hourly_weather_forecast"
NEIGHBORHOOD_URL = "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=500"


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard_data():
    engine = get_engine()
    with engine.connect() as connection:
        forecast = pd.read_sql(text(
            f"""
            Select * from {FORECAST_TABLE}
            """), con=connection)
        weather = pd.read_sql(text(
            f"""
            Select * from {WEATHER_TABLE} Order by hour
            """), con=connection)

    engine.dispose()

    forecast = forecast.sort_values(["hour","nta_code"])

    return forecast, weather


@st.cache_data(ttl=86_400, show_spinner=False)
def load_neighborhood_boundaries():
    boundaries = gpd.read_file(NEIGHBORHOOD_URL)
    boundaries = boundaries.rename(columns={"nta2020": "nta_code"})
    boundaries = boundaries[["nta_code", "geometry"]].copy()
    boundaries["nta_code"] = boundaries["nta_code"].astype(str)
    boundaries = boundaries.to_crs("EPSG:4326")

    return boundaries

def main():
    st.set_page_config(page_title= "NYC Citi-Bike Flow Forecast",
                    layout="wide")
    st.title("NYC Citi Bike neighborhood flow forecast")

    st.caption(
        "Explore predicted arrivals, departures, and net bike flow across "
        "NYC neighborhoods for the next seven days."
        )

    forecast, weather = load_dashboard_data()
    boundaries = load_neighborhood_boundaries()

    forecast_dates = sorted(forecast["hour"].dt.date.unique())
    selected_date = st.sidebar.selectbox("Forecast date", forecast_dates,
                                        format_func=lambda date: date.strftime("%a, %b %d"))

    date_rows = forecast.loc[forecast["hour"].dt.date == selected_date]
    selected_hour = st.sidebar.selectbox("Forecast hour", sorted(date_rows["hour"].unique()),
                                        format_func=lambda hour: pd.Timestamp(hour).strftime("%I:%M %p"))

    borough_options = sorted(forecast["borough"].unique())
    selected_boroughs = st.sidebar.multiselect("Borough", borough_options, default=borough_options)
    if not selected_boroughs:
        st.warning("Select at least one borough.")
        st.stop()

    color_by = st.sidebar.selectbox("Map color", ["Borough", "Net flow", "Arrivals", "Departures"])

    borough_forecast = forecast.loc[forecast["borough"].isin(selected_boroughs)]
    selected = borough_forecast.loc[borough_forecast["hour"] == pd.Timestamp(selected_hour)]
    if selected.empty:
        st.warning("No forecast rows match the selected filters.")
        st.stop()
        
    selected_time = pd.Timestamp(selected_hour)
    st.caption(f"Current selection: {selected_time:%b %d, %Y %I:%M %p}")

    largest_shortage = selected.nsmallest(1, "predicted_net_flow").iloc[0]
    largest_surplus = selected.nlargest(1, "predicted_net_flow").iloc[0]
    metric_columns = st.columns([1, 1, 2, 1])
    metric_columns[0].metric("Predicted Arrivals", f"{selected['predicted_arrivals'].sum():,.0f}")
    metric_columns[1].metric("Predicted Departures", f"{selected['predicted_departures'].sum():,.0f}")
    metric_columns[2].metric("Predicted Net Flow", f"{selected['predicted_net_flow'].sum():,.0f}")

    neighborhood_columns = st.columns(2)
    neighborhood_columns[0].metric("Largest Shortage",largest_shortage["nta_name"],
            f"{largest_shortage['predicted_net_flow']:+,.0f}",
            delta_color="off")

    neighborhood_columns[1].metric("Largest Surplus",largest_surplus["nta_name"],
                f"{largest_surplus['predicted_net_flow']:+,.0f}",
                delta_color="off")

    selected_weather = weather.loc[weather["hour"] == selected_time]
    weather_row = selected_weather.iloc[0]
    st.caption(
        "Selected-hour weather: "
        f"{weather_row['temperature']:.1f}°C. "
        f"{weather_row['precipitation']:.1f} mm precipitation. "
        f"{weather_row['wind_speed']:.1f} km/h wind"
    )

    st.subheader("Neighborhood map")
    st.plotly_chart(create_map(selected,boundaries,color_by))

    st.subheader("Forecast pattern")
    neighborhood_options = ["All selected neighborhoods"]+ sorted(borough_forecast["nta_name"].unique().tolist())

    selected_neighborhood = st.selectbox("Trend scope", neighborhood_options,
                                        index = neighborhood_options.index(largest_shortage["nta_name"]))

    if selected_neighborhood == "All selected neighborhoods":
        trend_data = borough_forecast
        trend_scope = ", ".join(selected_boroughs)
    else:
        trend_data = borough_forecast.loc[
            borough_forecast["nta_name"] == selected_neighborhood]
        
        trend_scope = selected_neighborhood

    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(create_demand_trend(trend_data, trend_scope))
    with right:
        st.plotly_chart(create_imbalance_ranking(selected))

    with st.expander("View selected-hour forecast data"):
        detail = selected[
            ["nta_name","borough","predicted_arrivals","predicted_departures","predicted_net_flow"]
            ].sort_values("predicted_net_flow")

        st.dataframe(detail, hide_index=True,
                    column_config={
                        "nta_name": "Neighborhood",
                        "borough": "Borough",
                        "predicted_arrivals": st.column_config.NumberColumn("Arrivals", format="%.1f"),
                        "predicted_departures": st.column_config.NumberColumn("Departures", format="%.1f"),
                        "predicted_net_flow": st.column_config.NumberColumn("Net flow", format="%+.1f")
                    })
        
    st.caption(
        "Sources: PostgreSQL hourly_neighborhood_forecast and "
        "hourly_weather_forecast; NYC Open Data 2020 NTA boundaries. "
        "Net flow equals predicted arrivals minus predicted departures"
    )

if __name__ =="__main__":
    main()
