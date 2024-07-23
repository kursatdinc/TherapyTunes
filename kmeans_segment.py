import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)

spotify_final = pd.read_csv("./datasets/spotify_final.csv")
spotify_final.head(10)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(spotify_final)

num_cols.append("mode")
num_cols.append("time_signature")

scaler = MinMaxScaler((0,1))
spotify_final[num_cols] = scaler.fit_transform(spotify_final[num_cols])

model_df = spotify_final[num_cols]

kmeans = KMeans(n_clusters=9 ,n_init=10).fit(model_df)

clusters_kmeans = kmeans.labels_

kmeans_df = spotify_final

kmeans_df["segment"] = clusters_kmeans

segment_df = kmeans_df[["track_id", "segment"]]

segment_df.to_csv("./datasets/segment.csv", index=False)

kmeans_df.head()


df_segment = pd.read_csv("./datasets/segment.csv")

df_segment[df_segment["segment"] == 0].sample(1)["track_id"].values[0]