import pandas as pd

pd.set_option("display.max_columns", None)

###########
mental_survey_df = pd.read_csv("./datasets/mental_survey_results.csv")
mental_survey_df.head(10)
###########

###########
mental_final = pd.read_csv("./datasets/mental_final.csv")
mental_final.head(10)
###########

###########
mental_after_eda = pd.read_csv("./datasets/mental_after_eda.csv")
mental_after_eda.head(10)
###########

###########
mental_before_ml = pd.read_csv("./datasets/mental_before_ml.csv")
mental_before_ml.head(10)
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
spotify_after_eda = pd.read_csv("./datasets/spotify_after_eda.csv")
spotify_after_eda.head(10)
###########
