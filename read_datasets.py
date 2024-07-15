import pandas as pd

pd.set_option('display.max_columns', None)

###########
mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
mental_survey_df.head(10)
###########

###########
lyrics_df = pd.read_csv("./datasets/lyrics_small.csv")
lyrics_df.head(10)
###########

###########
songs_df = pd.read_csv("./datasets/50000_songs.csv")
songs_df.head(10)
###########

###########
songs_df = songs_df.rename(columns={"name": "song"})
merged_df = pd.merge(lyrics_df, songs_df, how="inner", on="song")
merged_df.shape # (21015, 24)
merged_df.to_csv("./datasets/song_lyric.csv", index=False)

song_lyric_df = pd.read_csv("./datasets/song_lyric.csv")
###########

###########
combined_features_df = pd.read_csv("./datasets/combined_features_disorder.csv")
combined_features_df.drop(columns=["Unnamed: 0", "...1",
                                   "created_at", "type",
                                   "user_id"], axis=1, inplace=True)
combined_features_df.dropna(inplace=True)

combined_features_df.to_csv("./datasets/combined_features_disorder.csv", index=False)
###########