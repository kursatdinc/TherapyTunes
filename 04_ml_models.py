import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

mental_after_eda = pd.read_csv("./datasets/mental_after_eda.csv")

mental_after_eda.info()

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

cat_cols, num_cols, cat_but_car = grab_col_names(mental_after_eda)

num_cols = [col for col in num_cols if col not in ["anxiety", "depression", "insomnia", "tempo"]]

##############

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in mental_after_eda.columns if mental_after_eda[col].dtypes == "O" and mental_after_eda[col].nunique() == 2]

for col in binary_cols:
    mental_after_eda = label_encoder(mental_after_eda, col)

mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

frequency_columns = [col for col in mental_after_eda.columns if "frequency" in col]

for col in frequency_columns:
    mental_after_eda[col] = mental_after_eda[col].map(mapping)

mental_after_eda["music_effects"] = mental_after_eda["music_effects"].map({"Worsen": 0, "No Effect": 1, "Improve": 2})

##############

ohe_cols = ["streaming_service", "fav_genre"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dtype=int, drop_first=drop_first)
    return dataframe

mental_after_eda = one_hot_encoder(mental_after_eda, ohe_cols, drop_first=True)

mental_after_eda.head()

##############

scaler = RobustScaler()
mental_after_eda[num_cols] = scaler.fit_transform(mental_after_eda[num_cols])

##############

mental_after_eda.to_csv("./datasets/mental_before_ml.csv", index=False)

##############

mental_after_eda = pd.read_csv("./datasets/mental_before_ml.csv")

y = mental_after_eda["anxiety"]
X = mental_after_eda.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
##############

def weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, np.round(y_pred), weights="quadratic")

model = CatBoostClassifier(iterations=1000,
                           learning_rate=0.05,
                           depth=6,
                           loss_function="MultiClass",
                           eval_metric="WKappa")

model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          verbose=100)

y_pred = model.predict(X_test)

wkappa_score = weighted_kappa(y_test, y_pred)
print(f"Ağırlıklı Kappa Skoru: {wkappa_score}")

random_sample = X.sample(1)
prediction = model.predict(random_sample)

def plot_importance(model, features, dataframe, save=False):
    num = len(dataframe)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set_theme(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(model, X, mental_after_eda)