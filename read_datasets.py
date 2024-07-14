import pandas as pd

pd.set_option('display.max_columns', None)

mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
mental_survey_df.head(10)

lyrics_df = pd.read_csv("./datasets/lyrics_small.csv")
lyrics_df.head(10)

songs_df = pd.read_csv("./datasets/50000_songs.csv")
songs_df.head(10)

songs_df = songs_df.rename(columns={"name": "song"})

merged_df = pd.merge(lyrics_df, songs_df, how="inner", on="song")
merged_df.shape
# (21015, 24)