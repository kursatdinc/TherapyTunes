import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

##############
mental_final = pd.read_csv("./datasets/mental_before_ml.csv")
##############

y = mental_final["anxiety"]
X = mental_final.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

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

plot_importance(model, X, mental_final)