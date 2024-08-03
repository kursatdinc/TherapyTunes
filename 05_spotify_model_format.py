import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA
###########

df_spoti = pd.read_csv("./datasets/spotify_clustered.csv")

###########
# FEATURE EXTRACTION
###########

## ANXIETY INDEX

df_spoti["anxiety_index"] = (df_spoti["instrumentalness"] + (1 - (df_spoti["tempo"] / 200)) + df_spoti["valence"]) / 3


## DEPRESSION INDEX

df_spoti["depression_index"] = ((1 - df_spoti["valence"]) + (1 - df_spoti["danceability"]) + df_spoti["acousticness"]) / 3


## INSOMNIA INDEX

df_spoti["insomnia_index"] = ((1 - (df_spoti["energy"]) + df_spoti["acousticness"] + (1 - (df_spoti["loudness"] + 60) / 60))) / 3


###########
# ENCODING & SCALING
###########

minmax_cols = ["anxiety_index", "depression_index", "insomnia_index"]
scaler = MinMaxScaler()
df_spoti[minmax_cols] = scaler.fit_transform(df_spoti[minmax_cols])

df_spoti["valence"] = pd.qcut(df_spoti["valence"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
df_spoti["energy"] = pd.qcut(df_spoti["energy"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)


columns_to_keep = ["anxiety_index", "depression_index", "insomnia_index", "tempo", "valence", "energy", "cluster"]
df_spoti_model = df_spoti[columns_to_keep]


###########
# MODEL_DF EXPORT
###########

df_spoti_model.to_csv("./datasets/spotify_model.csv", index=False)