import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

##############
mental_final = pd.read_csv("./datasets/mental_final.csv")
##############

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

cat_cols, num_cols, cat_but_car = grab_col_names(mental_final)

num_cols = [col for col in num_cols if col not in ["anxiety", "depression", "insomnia", "obsession", "tempo"]]

##############

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in mental_final.columns if mental_final[col].dtypes == "O" and mental_final[col].nunique() == 2]

for col in binary_cols:
    mental_final = label_encoder(mental_final, col)

mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

frequency_columns = [col for col in mental_final.columns if "frequency" in col]

for col in frequency_columns:
    mental_final[col] = mental_final[col].map(mapping)

mental_final["music_effects"] = mental_final["music_effects"].map({"Worsen": 0, "No Effect": 1, "Improve": 2})

##############

ohe_cols = ["streaming_service", "fav_genre"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dtype=int, drop_first=drop_first)
    return dataframe

mental_final = one_hot_encoder(mental_final, ohe_cols, drop_first=True)

mental_final.head()

##############

scaler = RobustScaler()
mental_final[num_cols] = scaler.fit_transform(mental_final[num_cols])

##############

mental_final.to_csv("./datasets/mental_before_ml.csv", index=False)