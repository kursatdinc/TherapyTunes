import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.madf_surveycolumns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA
###########

df_survey = pd.read_csv("./datasets/mental_final.csv")


###########
# FEATURE ENGINEERING
###########

###################### FEATURE EXT ######################

# Feature 1
def get_age_group(age):
    if age >= 77:
        return 5
    elif age >= 59:
        return 4
    elif age >= 43:
        return 3
    elif age >= 27:
        return 2
    elif age >= 11:
        return 1
    else:
        return 0
        
df_survey["age_group"] = df_survey["age"].apply(get_age_group)
df_survey["age_group"] = df_survey["age_group"].astype(int)

# Feature 2
freq_cols = [col for col in df_survey.columns if "frequency" in col]

ordinal_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

for col in freq_cols:
    df_survey[col] = df_survey[col].map(ordinal_mapping)

df_survey["average_frequency"] = df_survey[freq_cols].mean(axis=1)

# Feature 3
def calculate_genre_diversity(row):
    non_zero_genres = sum(1 for value in row if value > 0)
    return non_zero_genres / len(freq_cols)
df_survey["genre_diversity"] = df_survey[freq_cols].apply(calculate_genre_diversity, axis=1)

# Feature 4
df_survey["normalized_hours"] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_survey[["hours_per_day"]])
df_survey["normalized_diversity"] = df_survey["genre_diversity"]
df_survey["normalized_frequency"] = df_survey["average_frequency"] / 3

df_survey["music_consumption_profile"] = (df_survey["normalized_hours"] * 0.3 +
                                          df_survey["normalized_diversity"] * 0.3 +
                                          df_survey["normalized_frequency"] * 0.4)

drop = ["normalized_hours", "normalized_diversity", "normalized_frequency"]
df_survey.drop(columns=drop, axis=1, inplace=True)

# Feature 5
df_survey["rock_metal_affinity"] = (df_survey["frequency_metal"] + df_survey["frequency_rock"] + 1) / 2
        
# Feature 6
df_survey["mainstream_music_score"] = df_survey["average_frequency"] * (1 - df_survey["genre_diversity"])

# Drop original columns that are no longer needed
df_survey = df_survey.drop(columns="average_frequency")

###################### ENCODING & SCALING ######################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dtype=int, drop_first=drop_first)
    return dataframe


binary_features = ["while_working", "instrumentalist", "exploratory", "music_effects"]

for col in binary_features:
    df_survey = label_encoder(df_survey, col)


categorical_features = ["streaming_service", "fav_genre"]

df_survey = one_hot_encoder(categorical_features, drop_first=True)

numeric_features = ["age", "age_group", "hours_per_day", "genre_diversity", "music_consumption_profile",
                    "rock_metal_affinity", "mainstream_music_score"]

frequency_features = ["frequency_instrumental", "frequency_traditional", "frequency_dance",
                      "frequency_jazz", "frequency_metal", "frequency_pop", "frequency_rnb",
                      "frequency_rap", "frequency_rock"]

scaler = StandardScaler()

df_survey[numeric_features] = scaler.fit_transform(df_survey[numeric_features])
df_survey[frequency_features] = scaler.fit_transform(df_survey[frequency_features])


###########
# MODELS
###########

def evaluate_models_with_grid_search(X, y):
    # StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Modeller ve hiperparametre gridleri
    classifiers = {
        "LR": (LogisticRegression(madf_surveyiter=1000), {
            "C": [0.1, 1, 10]
        }),
        "KNN": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }),
        "RFC": (RandomForestClassifier(), {
            "n_estimators": [50, 100],
            "criterion": ["gini", "entropy"],
            "madf_surveydepth": [None, 10, 20]
        }),
        "CBC": (CatBoostClassifier(silent=True), {
            "iterations": [100, 200],
            "learning_rate": [0.01, 0.1],
            "depth": [3, 6, 10]
        }),
        "XGB": (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "madf_surveydepth": [3, 6]
        }),
        "LGBM": (LGBMClassifier(verbose=-1), {
            "num_leaves": [31, 50],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [50, 100]
        }),
        "DT": (DecisionTreeClassifier(), {
            "criterion": ["gini", "entropy"],
            "madf_surveydepth": [None, 10, 20]
        }),
        "GBC": (GradientBoostingClassifier(), {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "madf_surveydepth": [3, 6]
        }),
        "ABC": (AdaBoostClassifier(), {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1]
        }),
        "SVC": (SVC(probability=True), {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "degree": [3, 5],
            "coef0": [0.0, 0.1]
        })
    }

    results = {}

    for model_name, (model, param_grid) in classifiers.items():
        print(f"\n{model_name} Modeli:")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_

        # Cross-validation kullanarak tahminleri al
        y_pred = cross_val_predict(best_model, X, y, cv=skf, method="predict")
        y_pred_proba = cross_val_predict(best_model, X, y, cv=skf, method="predict_proba")[:, 1]

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred_proba)

        # Sonuçları sakla
        results[model_name] = {
            "Best Parameters": grid_search.best_params_,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Recall": recall,
            "Precision": precision,
            "ROC AUC": roc_auc
        }

    # Sonuçları yazdır
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"Best Parameters: {metrics['Best Parameters']}")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"F1 Score: {metrics['F1 Score']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"ROC AUC: {metrics['ROC AUC']:.4f}")



###################### TEMPO ######################

y = df_survey["tempo"]
X = df_survey.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

df_surveytrain, df_surveytest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_xgb_with_feature_selection(X, y, n_splits=5, param_grid=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pipeline oluştur
    pipeline = Pipeline([
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("model", XGBRegressor(
            learning_rate=0.01,
            madf_surveydepth=3,
            n_estimators=100,
            subsample=0.8,
            random_state=42
        ))
    ])

    # Hiperparametre grid"i
    if param_grid is None:
        param_grid = {"feature_selection__k": [5, 10, 15]}

        # GridSearchCV ile en iyi parametreleri bulma
        grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        df_surveytrain, df_surveytest = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model.fit(df_surveytrain, y_train)
        y_pred = best_model.predict(df_surveytest)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    print(f"En İyi Parametreler: {grid_search.best_params_}")
    print(f"Ortalama MSE: {np.mean(mse_scores):.4f}")
    print(f"Ortalama RMSE: {np.sqrt(np.mean(mse_scores)):.4f}")
    print(f"Ortalama R^2: {np.mean(r2_scores):.4f}")

    # Seçilen Özellikler
    selected_features = X.columns[best_model.named_steps["feature_selection"].get_support()]
    print("\nSeçilen Özellikler:")
    print(selected_features)

evaluate_xgb_with_feature_selection(X, y)



###################### ANXIETY ######################

y = df_survey["anxiety"]
X = df_survey.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

df_surveytrain, df_surveytest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_abc_with_feature_selection(X, y, best_params, n_splits=5, k_features=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Özellik seçim fonksiyonu
    feature_selector = SelectKBest(score_func=f_classif, k=k_features)

    # Modeli kurma
    abc_model = AdaBoostClassifier(learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"],
                                   random_state=42)

    # Pipeline oluşturma
    pipeline = Pipeline([
        ("feature_selection", feature_selector),
        ("classification", abc_model)
    ])

    # Cross-validation kullanarak tahminleri al
    y_pred = cross_val_predict(pipeline, X, y, cv=skf, method="predict")
    y_pred_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Sonuçları yazdır
    print("AdaBoostClassifier Results with Feature Selection and Optimized Parameters using 5-Fold Cross-Validation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

best_params = {"learning_rate": 0.1, "n_estimators": 100}

evaluate_abc_with_feature_selection(X, y, best_params)

#Accuracy: 0.57
#F1 Score: 0.58
#Recall: 0.61
#Precision: 0.55
#ROC AUC: 0.59



###################### DEPRESSION ######################

y = df_survey["depression"]
X = df_survey.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

df_surveytrain, df_surveytest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_svc_with_kfold(X, y, best_params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Modeli kur
    svc_model = SVC(
        C=best_params["C"],
        kernel=best_params["kernel"],
        degree=best_params["degree"],
        coef0=best_params["coef0"],
        probability=True,
        random_state=42
    )

    # Cross-validation kullanarak tahminleri al
    y_pred = cross_val_predict(svc_model, X, y, cv=skf, method="predict")
    y_pred_proba = cross_val_predict(svc_model, X, y, cv=skf, method="predict_proba")[:, 1]

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Sonuçları yazdır
    print("SVC Results with Optimized Parameters using 5-Fold Cross-Validation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

best_params_svc = {"C": 1, "kernel": "rbf", "degree": 3, "coef0": 0.0}

evaluate_svc_with_kfold(X, y, best_params_svc)

#Accuracy: 0.61
#F1 Score: 0.56
#Recall: 0.53
#Precision: 0.60
#ROC AUC: 0.65



###################### INSOMNIA ######################

y = df_survey["insomnia"]
X = df_survey.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

df_surveytrain, df_surveytest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_rfc_with_feature_selection(X, y, best_params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Özelliklerin seçilmesi için model kurulumu
    feature_selector = SelectFromModel(RandomForestClassifier(
        criterion=best_params["criterion"],
        madf_surveydepth=best_params["madf_surveydepth"],
        n_estimators=best_params["n_estimators"],
        random_state=42
    ))

    # RandomForestClassifier modelini kur
    rfc_model = RandomForestClassifier(
        criterion=best_params["criterion"],
        madf_surveydepth=best_params["madf_surveydepth"],
        n_estimators=best_params["n_estimators"],
        random_state=42
    )

    # Pipeline kur
    pipeline = Pipeline([
        ("feature_selection", feature_selector),
        ("model", rfc_model)
    ])

    # Cross-validation kullanarak tahminleri al
    y_pred = cross_val_predict(pipeline, X, y, cv=skf, method="predict")
    y_pred_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Sonuçları yazdır
    print("RandomForestClassifier with Feature Selection Results using 5-Fold Cross-Validation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Seçilen Özellikler
    pipeline.fit(X, y)  # Model ve özellik seçici fit edilmelidir
    feature_selector = pipeline.named_steps["feature_selection"]
    selected_features = X.columns[feature_selector.get_support()]
    print("\nSeçilen Özellikler:")
    print(selected_features)

best_params = {"criterion": "gini", "madf_surveydepth": None, "n_estimators": 100}

evaluate_rfc_with_feature_selection(X, y, best_params)

#Accuracy: 0.58
#F1 Score: 0.53
#Recall: 0.50
#Precision: 0.56
#ROC AUC: 0.63