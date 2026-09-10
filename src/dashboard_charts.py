import json
import numpy as np
import pandas as pd
import plotly.express as px


def create_map(selected, boundaries, color_by):
    map_data = boundaries.merge(selected, on="nta_code",how="inner")

    geojson = json.loads(map_data[["nta_code","geometry"]].to_json())
    plot_data = pd.DataFrame(map_data.drop(columns="geometry"))

    map_options ={
        "data_frame": plot_data,
        "geojson": geojson,
        "locations": "nta_code",
        "featureidkey": "properties.nta_code",
        "hover_name": "nta_name",
        "hover_data": {
            "nta_code": False,
            "borough": True,
            "predicted_arrivals": ":,.1f",
            "predicted_departures": ":,.1f",
            "predicted_net_flow": ":+,.1f"
        },
        "center": {"lat": 40.718, "lon": -74},
        "zoom": 9.5,
        "opacity": 0.69,
        "map_style": "carto-positron",
        "height": 550
    }

    if color_by =="Borough":
        fig = px.choropleth_map(**map_options, color="borough",
                                color_discrete_sequence=px.colors.qualitative.Set1,
                                labels={"borough": "Borough"})

    else:
        metric = {
            "Net flow": "predicted_net_flow",
            "Arrivals": "predicted_arrivals",
            "Departures": "predicted_departures"}[color_by]

        if color_by == "Net flow":
            largest_flow = max(float(plot_data["predicted_net_flow"].abs().max()), 1.0)
            fig = px.choropleth_map(**map_options,color=metric,
                                    color_continuous_scale=["#FF2600", "#FFFFFF", "#84FF00"],
                                    range_color=(-largest_flow, largest_flow),
                                    labels={metric: "Net flow"})
        else:
            if color_by=="Arrivals":
                color_scale = "Blues"
            else:
                color_scale="Oranges"
            fig = px.choropleth_map(**map_options,color=metric,
                                    color_continuous_scale=color_scale,
                                    range_color=(0, float(plot_data[metric].max())),
                                    labels={metric: color_by})
    fig.update_traces(marker_line_width=0.5, marker_line_color="#000208")
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, 
                    legend_title_text="Borough")

    return fig



def create_demand_trend(data, scope_name):
    hourly_flow = (data.groupby("hour",as_index=False)[["predicted_arrivals", "predicted_departures"]]
            .sum()
            .rename(columns={
                "predicted_arrivals": "Arrivals",
                "predicted_departures": "Departures"})
            )

    trend_data = hourly_flow.melt(
        id_vars="hour",
        value_vars=["Arrivals", "Departures"],
        var_name="Flow type",
        value_name="Predicted rides"
    )

    fig = px.line(
        trend_data,
        x="hour",
        y="Predicted rides",
        color="Flow type",
        color_discrete_map={
            "Arrivals": "#83FD00",
            "Departures": "#FF2600"
        },
        title=f"Hourly predicted arrivals and departures: {scope_name}"
    )
    fig.update_layout(
        height=420,
        xaxis_title=None,
        yaxis_title="Predicted rides",
        legend_title_text=None,
        hovermode="x unified"
    )
    fig.update_yaxes(rangemode="tozero")

    return fig


def create_imbalance_ranking(selected):
    shortages = selected.nsmallest(5, "predicted_net_flow")
    surpluses = selected.nlargest(5, "predicted_net_flow")
    ranking = pd.concat([shortages, surpluses]).drop_duplicates("nta_code")
    ranking = ranking.sort_values("predicted_net_flow")
    ranking["Direction"] = np.where(
        ranking["predicted_net_flow"] < 0,
        "Shortage",
        "Surplus"
    )

    fig = px.bar(
        ranking,
        x="predicted_net_flow",
        y="nta_name",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Shortage": "#FF2600",
            "Surplus": "#83FD00"
        },
        text_auto=".2f",
        labels={
            "predicted_net_flow": "Predicted net flow",
            "nta_name": "Neighborhood"
        },
        title=f"Largest neighborhood imbalances"
    )
    fig.add_vline(x=0, line_color="#000000", line_width=0.5)
    fig.update_layout(height=450, yaxis_title=None, legend_title_text=None)

    return fig


