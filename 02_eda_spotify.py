import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Adjusting Row Column Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA - SPOTIFY
###########
df_spoti = pd.read_csv("./datasets/spotify_data.csv")

###########
# GENRE MATCH
###########

genre_drop_list = ["songwriter", "romance", "detroit-techno", "chicago-house", "grindcore", "party", "show-tunes", "dubstep", "house", 
                   "drum-and-bass", "trance","minimal-techno", "sad", "progressive-house", "comedy"]

df_spoti = df_spoti[~df_spoti["genre"].isin(genre_drop_list)]


inst_genres = ["acoustic", "ambient", "classical", "electronic", "opera", "piano", 
               "sleep", "guitar", "new-age"]

df_spoti.loc[df_spoti["genre"].isin(inst_genres), "genre"] = "Instrumental"

###################################
dance_genres = ["deep-house", "disco", "dub", "edm", "electro", "electronic",
    "dance", "techno", "dancehall", "garage", "hardstyle", "club", "salsa", "samba"]
df_spoti.loc[df_spoti["genre"].isin(dance_genres), "genre"] = "Dance"
###################################
tra_genres = ["country", "folk", "tango", "flamenco","spanish",
              "french","german","swedish", "indian", "sertanejo", "ska", "forro"]
df_spoti.loc[df_spoti["genre"].isin(tra_genres), "genre"] = "Traditional"
###################################
rap_genres = ["hip-hop", "breakbeat", "funk" ]
df_spoti.loc[df_spoti["genre"].isin(rap_genres), "genre"] = "Rap"
###################################
rb_genres = ["soul", "trip-hop", "blues", "gospel"]
df_spoti.loc[df_spoti["genre"].isin(rb_genres), "genre"] = "R&B"
###################################
rock_genres = ["alt-rock", "hard-rock","punk-rock", "rock","rock-n-roll",
    "psych-rock","indie-rock", "punk", "goth", "emo", "industrial"]
df_spoti.loc[df_spoti["genre"].isin(rock_genres), "genre"] = "Rock"
###################################
metal_genres = ["black-metal", "death-metal", "heavy-metal", "metal", "metalcore", "hardcore"]
df_spoti.loc[df_spoti["genre"].isin(metal_genres), "genre"] = "Metal"
####################################
pop_genres = ["pop","pop-film", "power-pop", "indie-pop", "cantopop", "k-pop", "singer-songwriter"]
df_spoti.loc[df_spoti["genre"].isin(pop_genres), "genre"] = "Pop"
#####################################
jazz_genres = ["afrobeat", "chill", "groove",  "jazz"]
df_spoti.loc[df_spoti["genre"].isin(jazz_genres), "genre"] = "Jazz"

###########
# MISSING VALUES & OUTLIERS
###########

## Tempo min 40 // max 250
df_spoti["tempo"] = df_spoti["tempo"].clip(lower=40, upper=250)

# Function to calculate lower and upper thresholds
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Function to check for outliers in a specific column
def check_outlier(dataframe, col_name):
    if pd.api.types.is_numeric_dtype(dataframe[col_name]):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)][col_name]
    else:
        return pd.Series([])

# Function to replace outliers with defined thresholds
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Selecting all numeric columns
numeric_vars = df_spoti.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Iterating through each numeric column to check and handle outliers
for col in numeric_vars:
    outliers = check_outlier(df_spoti, col)
    if not outliers.empty:
        print(f"Outliers found in {col}. Handling outliers...")
        replace_with_thresholds(df_spoti, col)

print("Outlier handling completed.")


###########
# EXPORT DATA
###########

df_spoti.to_csv("./datasets/spotify_final.csv", index=False)
