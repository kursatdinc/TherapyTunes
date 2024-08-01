import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA
###########

df_spoti = pd.read_csv("./datasets/spotify_final.csv")

###########
# K-MEANS CLUSTERING
###########

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

cat_cols, num_cols, cat_but_car = grab_col_names(df_spoti)

num_cols = [col for col in num_cols if col not in ["popularity", "year"]]
num_cols.append("mode")
num_cols.append("time_signature")

model_df = df_spoti[num_cols]

sc = MinMaxScaler((0, 1))
model_df[num_cols] = sc.fit_transform(model_df[num_cols])

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 30))

elbow.fit(model_df)
elbow.show()

kmeans = KMeans(n_clusters=elbow.elbow_value_,n_init=50).fit(model_df)

clusters_kmeans = kmeans.labels_

clustered_spoti_df = df_spoti
clustered_spoti_df["cluster"] = clusters_kmeans 

clustered_spoti_df.head()

###########
# RULE BASED CLUSTERING
###########

pca_cols = ['speechiness',
            'acousticness',
            'instrumentalness',
            'liveness',
            'valence',
            'tempo']

pca_df = df_spoti[pca_cols]

pca_df = StandardScaler().fit_transform(pca_df)

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(pca_df)

np.cumsum(pca.explained_variance_ratio_)

pca_df_ =pd.DataFrame(pca_fit, columns=["pc1", "pc2"])

pca_df_["pc1_score"] = pd.qcut(pca_df_["pc1"], 3, labels=[3, 2, 1])
pca_df_["pc2_score"] = pd.qcut(pca_df_["pc2"], 3, labels=[3, 2, 1])

pca_df_["pc_segment"] = pca_df_["pc1_score"].astype(str) + pca_df_["pc2_score"].astype(str)

###########

final_df = pd.concat([clustered_spoti_df, pca_df_], axis=1)
final_df.drop(columns=["pc1", "pc2", "pc1_score", "pc2_score"], inplace=True)
final_df.head()

###########
# FINAL_DF EXPORT
###########

final_df.to_csv("./datasets/spotify_clustered.csv", index=False)

###########
# SEGMENT EXPORTS
###########

df_list_11 = final_df[final_df["pc_segment"] == "11"].sort_values("popularity", ascending=False).iloc[:100]
df_list_11 = df_list_11[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_11.to_csv("./segment_datasets/segment_11.csv", index=False)

df_list_12 = final_df[final_df["pc_segment"] == "12"].sort_values("popularity", ascending=False).iloc[:100]
df_list_12 = df_list_12[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_12.to_csv("./segment_datasets/segment_12.csv", index=False)

df_list_13 = final_df[final_df["pc_segment"] == "13"].sort_values("popularity", ascending=False).iloc[:100]
df_list_13 = df_list_13[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_13.to_csv("./segment_datasets/segment_13.csv", index=False)

df_list_21 = final_df[final_df["pc_segment"] == "21"].sort_values("popularity", ascending=False).iloc[:100]
df_list_21 = df_list_21[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_21.to_csv("./segment_datasets/segment_21.csv", index=False)

df_list_22 = final_df[final_df["pc_segment"] == "22"].sort_values("popularity", ascending=False).iloc[:100]
df_list_22 = df_list_22[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_22.to_csv("./segment_datasets/segment_22.csv", index=False)

df_list_23 = final_df[final_df["pc_segment"] == "23"].sort_values("popularity", ascending=False).iloc[:100]
df_list_23 = df_list_23[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_23.to_csv("./segment_datasets/segment_23.csv", index=False)

df_list_31 = final_df[final_df["pc_segment"] == "31"].sort_values("popularity", ascending=False).iloc[:100]
df_list_31 = df_list_31[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_31.to_csv("./segment_datasets/segment_31.csv", index=False)

df_list_32 = final_df[final_df["pc_segment"] == "32"].sort_values("popularity", ascending=False).iloc[:100]
df_list_32 = df_list_32[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_32.to_csv("./segment_datasets/segment_32.csv", index=False)

df_list_33 = final_df[final_df["pc_segment"] == "33"].sort_values("popularity", ascending=False).iloc[:100]
df_list_33 = df_list_33[["artist_name", "track_name", "track_id"]].reset_index(drop=True)
df_list_33.to_csv("./segment_datasets/segment_33.csv", index=False)