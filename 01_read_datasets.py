import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
mental_survey_df.head(10)
###########

###########
mental_final = pd.read_csv("./datasets/mental_final.csv")
mental_final.head(10)
###########

###########
###########

###########
spotify_df = pd.read_csv("./datasets/spotify_data.csv")
spotify_df.head(10)
###########

###########
spotify_final = pd.read_csv("./datasets/spotify_final.csv")
spotify_final.head(10)
###########

###########
spotify_clustered = pd.read_csv("./datasets/spotify_clustered.csv")
spotify_clustered.head(10)
###########

###########
spotify_model = pd.read_csv("./datasets/spotify_model.csv")
spotify_model.head(10)
###########