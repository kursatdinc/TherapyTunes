import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

song_lyric_df = pd.read_csv("./datasets/song_lyric.csv")

def polar_plot(df, track_id):
    labels = ["energy", "danceability", "acousticness", "valence",
                 "speechiness", "liveness", "instrumentalness"]
    
    features = df[df["track_id"] == track_id][labels]

    stats = features.iloc[0].to_list()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    stats += stats[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, stats, color='grey', alpha=0.25)
    ax.plot(angles, stats, 'o-', color='grey', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.show()

polar_plot(song_lyric_df, "TRFZRQD128F428B23F")