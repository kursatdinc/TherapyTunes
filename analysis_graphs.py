import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
                                   paper_bgcolor = "#E8E8E8",
                                   title=f"Polar Plot of {genre.capitalize()} Features")

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
                                   paper_bgcolor = "#E8E8E8",
                                   title=f"Polar Plot of {artist_name}'s Features")

    fig.update_polars(angularaxis = dict(ticktext = [f"<b>{label.capitalize()}</b>" for label in labels],
                                         tickvals = angles[:-1],
                                         tickfont = dict(size=12)))

    st.plotly_chart(fig)


def genres_by_years(df):
    df = df[df["year"] <= 2022]
    
    genre_years = df.groupby(["year", "genre"]).size().reset_index(name="count")

    fig = px.line(
        genre_years,
        x="year",
        y="count",
        color="genre",
        title="Distribution of Music Genres Over Years",
        labels={"year": "Year", "count": "Number of Tracks"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Tracks",
        legend_title="Music Genre",
        legend=dict(
            title="Music Genre",
            x=1,
            y=1
        ),
        plot_bgcolor="#E8E8E8",
        paper_bgcolor="#E8E8E8",
        xaxis=dict(
            rangeslider=dict(
                visible=True
            )
        )
    )
    
    st.plotly_chart(fig)


def genre_popularity(df):
    genre_popularity = df.groupby("genre")["popularity"].mean().reset_index()

    y_min = 15
    y_max = 40


    fig = px.bar(
        genre_popularity,
        x="genre",
        y="popularity",
        title="Music Genre - Average Popularity",
        labels={"genre": "Music Genre", "popularity": "Average Popularity"},
        color="popularity",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        yaxis=dict(range=[y_min, y_max]),
        plot_bgcolor="#E8E8E8",
        paper_bgcolor="#E8E8E8",
        showlegend = False
    )

    st.plotly_chart(fig)


def top_songs(df,n):
    top_songs = df.nlargest(n, "popularity")

    y_min = 70
    y_max = 100

    fig = px.bar(
        top_songs,
        x="track_name",
        y="popularity",
        title=f"Top {n} Most Popular Songs",
        labels={"track_name": "Song Name", "popularity": "Popularity"},
        color="popularity",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(xaxis_title="Song Name",
                      yaxis_title="Popularity",
                      yaxis=dict(range=[y_min, y_max]),
                      plot_bgcolor="#E8E8E8",
                      paper_bgcolor="#E8E8E8",
                      showlegend = False
    )
    
    st.plotly_chart(fig)


def tempo_by_genre(df):
    genre_avg_tempo = df.groupby("genre")["tempo"].mean().reset_index()

    y_min = 100
    y_max = 130

    fig = px.bar(
        genre_avg_tempo,
        x="genre",
        y="tempo",
        title="Average Tempo by Music Genre",
        labels={"genre": "Music Genre", "tempo": "Average Tempo"},
        color="genre"
    )

    fig.update_layout(
        xaxis_title="Music Genre",
        yaxis_title="Average Tempo",
        yaxis=dict(range=[y_min, y_max]),
        plot_bgcolor="#E8E8E8",
        paper_bgcolor="#E8E8E8",
        showlegend=False
    )
    
    st.plotly_chart(fig)


def d_stage(df):
    data = df.groupby(["year", "genre"]).size().reset_index(name="count")
    
    feature_data = data.pivot(index="year", columns="genre", values="count")

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=feature_data.values,
        x=feature_data.columns,
        y=feature_data.index,
        colorscale="Viridis"
    ))

    fig.update_layout(
        title="Distribution of Music Genres Over Years",
        scene=dict(
            xaxis_title="Genre",
            yaxis_title="Year",
            zaxis_title="Count"
        ),
        plot_bgcolor="#E8E8E8",
        paper_bgcolor="#E8E8E8",
        showlegend = False
    )

    st.plotly_chart(fig)


def mental_health_by_music(df):
    psych_data = df.groupby("fav_genre").agg({
    "anxiety": "mean",
    "depression": "mean",
    "insomnia": "mean"
    }).reset_index()

    fig = px.bar(
        psych_data,
        x="fav_genre",
        y=["anxiety", "depression", "insomnia"],
        title="Mental Healths According to Music Genres",
        labels={"fav_genre": "Music Genre", "value": "Average Level"},
        barmode="group"
    )

    fig.update_layout(xaxis_title="Music Genre",
                      yaxis_title="Average Level",
                      plot_bgcolor="#E8E8E8",
                      paper_bgcolor="#E8E8E8",
                      showlegend = True,
                      legend_title_text="Mental Healths")

    st.plotly_chart(fig)


def genre_usage(df):
    genre_usage = df[["fav_genre"]].value_counts(normalize=True).reset_index(name="Percentage")

    fig = px.pie(
        genre_usage,
        names="fav_genre",
        values="Percentage",
        title="Fav Genre Distrubition"
    )

    fig.update_layout(plot_bgcolor="#E8E8E8",
                      paper_bgcolor="#E8E8E8",
                      showlegend = True)

    st.plotly_chart(fig)


def age_genre_dist(df):
    age_genre_data = df.groupby(["age", "fav_genre"]).size().reset_index(name="count")

    fig = px.bar(
        age_genre_data,
        x="age",
        y="count",
        color="fav_genre",
        title="Age-Fav Genre Distribution",
        labels={"age": "Age", "count": "Number of Users"}
    )

    fig.update_layout(xaxis_title="Age",
                      yaxis_title="Number of Users",
                      plot_bgcolor="#E8E8E8",
                      paper_bgcolor="#E8E8E8",
                      showlegend = True,
                      legend_title_text="Fav Genre")
    
    st.plotly_chart(fig)


def genre_hour(df):
    fig = px.box(
        df, 
        x="fav_genre", 
        y="hours_per_day", 
        title="Hours per Day by Favorite Genre")
    
    fig.update_layout(
            xaxis_title="Fav Genre",
            yaxis_title="Hours per Day",
            plot_bgcolor="#E8E8E8",
            paper_bgcolor="#E8E8E8",
            showlegend = True)
    
    st.plotly_chart(fig)