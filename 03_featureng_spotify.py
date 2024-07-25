import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

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

pca_df_ =pd.DataFrame(pca_fit, columns=["PC1", "PC2"])

pca_df_["PC1_Score"] = pd.qcut(pca_df_["PC1"], 2, labels=[2, 1])
pca_df_["PC2_Score"] = pd.qcut(pca_df_["PC2"], 2, labels=[2, 1])

pca_df_["PC_Segment"] = pca_df_["PC1_Score"].astype(str) + pca_df_["PC2_Score"].astype(str)

###########

final_df = pd.concat([clustered_spoti_df, pca_df_], axis=1)
final_df.head()

###########
# FEATURE EXTRACTION
###########


###########
# ENCODING & SCALING
###########


##############
##############
##############

###########
# BASE MODELS
###########