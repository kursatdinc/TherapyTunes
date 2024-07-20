import pandas as pd
import numpy as np
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

y = mental_after_eda["anxiety"]
X = mental_after_eda.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class QWKObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        labels = np.array(targets) + a
        preds = np.array(approxes) + a
        preds = preds.clip(1, 6)
        f = 1/2 * np.sum((preds - labels)**2)
        g = 1/2 * np.sum((preds - a)**2 + b)
        df = preds - labels
        dg = preds - a
        grad = (df / g - f * dg / g**2) * len(labels)
        hess = (1 / g - (2 * df * dg) / (g**2) + (2 * f * dg**2) / (g**3)) * len(labels)
        return list(zip(grad, hess))

class QWKMetric(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, targets, weight):
        approxes = approxes[0]
        targets = np.array(targets) + a
        approxes = np.array(approxes) + a
        approxes = approxes.clip(1, 6).round()
        qwk = cohen_kappa_score(targets, approxes, weights="quadratic")
        return qwk, 0
    
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