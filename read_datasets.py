import pandas as pd

pd.set_option('display.max_columns', None)

###########
mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
mental_survey_df.head(10)
###########

###########
spotify_df = pd.read_csv("./datasets/spotify_data.csv")
spotify_df.head(10)
###########