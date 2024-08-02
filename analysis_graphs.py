import streamlit as st
import plotly.graph_objects as go
import numpy as np


def polar_plot(df, genre):
    labels = ["energy", "danceability", "acousticness", "valence",
              "speechiness", "liveness", "instrumentalness"]
    
    features = df[df["genre"] == genre][labels].mean()

    stats = features.to_list()

    angles = np.linspace(0, 360, len(labels), endpoint=False)

    stats += stats[:1]
    angles = np.append(angles, angles[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(r = stats,
                                  theta = angles,
                                  fill = "toself",
                                  fillcolor = "rgba(255, 0, 0, 0.3)",
                                  line = dict(color = "red"),
                                  name = genre))

    fig.update_layout(polar = dict(radialaxis = dict(visible = True,
                                                     range = [0, 1],
                                                     showticklabels = False,
                                                     ticks = "",),
                                   bgcolor = "#E8E8E8",),
                                   showlegend = False,
                                   paper_bgcolor = "#E8E8E8")

    fig.update_polars(angularaxis = dict(ticktext = [f"<b>{label.capitalize()}</b>" for label in labels],
                                         tickvals = angles[:-1],
                                         tickfont = dict(size=12)))

    st.plotly_chart(fig)


def artist_radar_plot(df, artist_name):

    labels = ["energy", "danceability", "acousticness", "valence",
              "speechiness", "liveness", "instrumentalness"]

    features = df[df["artist_name"] == artist_name][labels].mean()

    stats = features.to_list()

    angles = np.linspace(0, 360, len(labels), endpoint=False)

    stats += stats[:1]
    angles = np.append(angles, angles[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(r = stats,
                                  theta = angles,
                                  fill = "toself",
                                  fillcolor = "rgba(255, 0, 0, 0.3)",
                                  line = dict(color = "red"),
                                  name = artist_name))

    fig.update_layout(polar = dict(radialaxis = dict(visible = True,
                                                     range = [0, 1],
                                                     showticklabels = False,
                                                     ticks = "",),
                                   bgcolor = "#E8E8E8",),
                                   showlegend = False,
                                   paper_bgcolor = "#E8E8E8")

    fig.update_polars(angularaxis = dict(ticktext = [f"<b>{label.capitalize()}</b>" for label in labels],
                                         tickvals = angles[:-1],
                                         tickfont = dict(size=12)))

    st.plotly_chart(fig)