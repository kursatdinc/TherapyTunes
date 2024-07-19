import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

###########
mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
mental_survey_df.head(10)
###########

###########
spotify_df = pd.read_csv("./datasets/spotify_data.csv", index=False)
spotify_df.head(10)
###########