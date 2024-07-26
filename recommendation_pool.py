import pandas as pd

pd.set_option("display.max_columns", None)

###########
df = pd.read_csv("./datasets/spotify_final.csv")
df.head(10)
###########

recom_pool = df[df["cluster"] = answer_dict.get("cluster") and df["pc_segment"] = answer_dict.get("pc_segment") and df["genre"] = answer_dict.get("fav_genre")].iloc[:100]

sample = recom_pool.sample(1)["track_id"]