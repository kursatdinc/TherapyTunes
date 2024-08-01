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

df_spoti["anxiety_index"] = df_spoti["danceability"]*5 + df_spoti["loudness"]/10 + df_spoti["tempo"]/20 - df_spoti["acousticness"]*2
df_spoti["anxiety_index"] = pd.qcut(df_spoti["anxiety_index"], 2, labels=[1, 0])

## DEPRESSION INDEX

df_spoti["depression_index"] = 10 - df_spoti["valence"]*7 + df_spoti["energy"]*3 + (1-df_spoti["mode"])*2
df_spoti["depression_index"] = pd.qcut(df_spoti["depression_index"], 2, labels=[1, 0])

## INSOMNIA INDEX

df_spoti["insomnia_index"] = (1 - df_spoti["acousticness"])*3 + df_spoti["energy"]*4 + df_spoti["loudness"]/15 + (1 - df_spoti["instrumentalness"])*3
df_spoti["insomnia_index"] = pd.qcut(df_spoti["insomnia_index"], 2, labels=[1, 0])

## OBSESSION INDEX

df_spoti["obsession_index"] = df_spoti["energy"]*5 + df_spoti["loudness"]/10 + (1 - df_spoti["valence"])*5
df_spoti["obsession_index"] = pd.qcut(df_spoti["insomnia_index"], 2, labels=[1, 0])

###########
# ENCODING & SCALING
###########


##############
##############
##############

###########
# BASE MODELS
###########