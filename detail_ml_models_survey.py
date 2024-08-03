##################### Import Libraries ####################
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, StratifiedKFold, cross_val_predict)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")

##################### COLUMN SETTINGS ####################
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_survey = pd.read_csv("./datasets/mental_final.csv")


def evaluate_models_with_grid_search(X, y):
    # StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Modeller ve hiperparametre gridleri
    classifiers = {
        "LR": (LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10]
        }),
        "KNN": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),
        "RFC": (RandomForestClassifier(), {
            'n_estimators': [50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20]
        }),
        "CBC": (CatBoostClassifier(silent=True), {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1],
            'depth': [3, 6, 10]
        }),
        "XGB": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }),
        "LGBM": (LGBMClassifier(verbose=-1), {
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100]
        }),
        "DT": (DecisionTreeClassifier(), {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20]
        }),
        "GBC": (GradientBoostingClassifier(), {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }),
        "ABC": (AdaBoostClassifier(), {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }),
        "SVC": (SVC(probability=True), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'degree': [3, 5],
            'coef0': [0.0, 0.1]
        })
    }

    results = {}

    for model_name, (model, param_grid) in classifiers.items():
        print(f"\n{model_name} Modeli:")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
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






## ANXIETY MODEL ##

y = df_survey["Anxiety"]
X = df_survey.drop(columns=["BPM", "Anxiety", "Depression", "Insomnia"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_abc_with_feature_selection(X, y, best_params, n_splits=5, k_features=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Özellik seçim fonksiyonu
    feature_selector = SelectKBest(score_func=f_classif, k=k_features)

    # Modeli kurma
    abc_model = AdaBoostClassifier(learning_rate=best_params['learning_rate'], n_estimators=best_params['n_estimators'],
                                   random_state=42)

    # Pipeline oluşturma
    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('classification', abc_model)
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
best_params = {'learning_rate': 0.1, 'n_estimators': 100}
evaluate_abc_with_feature_selection(X, y, best_params)

#Accuracy: 0.57
#F1 Score: 0.58
#Recall: 0.61
#Precision: 0.55
#ROC AUC: 0.59







## DEPRESSION MODEL ##

y = df_survey["Depression"]
X = df_survey.drop(columns=["BPM", "Anxiety", "Depression", "Insomnia"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_svc_with_kfold(X, y, best_params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Modeli kur
    svc_model = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        degree=best_params['degree'],
        coef0=best_params['coef0'],
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
best_params_svc = {'C': 1, 'kernel': 'rbf', 'degree': 3, 'coef0': 0.0}
evaluate_svc_with_kfold(X, y, best_params_svc)


#Accuracy: 0.61
#F1 Score: 0.56
#Recall: 0.53
#Precision: 0.60
#ROC AUC: 0.65







## INSOMNIA MODEL ##

y = df_survey["Insomnia"]
X = df_survey.drop(columns=["BPM", "Anxiety", "Depression", "Insomnia"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



def evaluate_rfc_with_feature_selection(X, y, best_params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Özelliklerin seçilmesi için model kurulumu
    feature_selector = SelectFromModel(RandomForestClassifier(
        criterion=best_params['criterion'],
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        random_state=42
    ))

    # RandomForestClassifier modelini kur
    rfc_model = RandomForestClassifier(
        criterion=best_params['criterion'],
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        random_state=42
    )

    # Pipeline kur
    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('model', rfc_model)
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
    feature_selector = pipeline.named_steps['feature_selection']
    selected_features = X.columns[feature_selector.get_support()]
    print("\nSeçilen Özellikler:")
    print(selected_features)
best_params = {
    'criterion': 'gini',
    'max_depth': None,
    'n_estimators': 100
}
evaluate_rfc_with_feature_selection(X, y, best_params)


#Accuracy: 0.58
#F1 Score: 0.53
#Recall: 0.50
#Precision: 0.56
#ROC AUC: 0.63





## TEMPO MODEL ##

y = df_survey["BPM"]
X = df_survey.drop(columns=["BPM", "Anxiety", "Depression", "Insomnia"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


################### XGBOOST ###############
def evaluate_xgb_with_feature_selection(X, y, n_splits=5, param_grid=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pipeline oluştur
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('model', XGBRegressor(
            learning_rate=0.01,
            max_depth=3,
            n_estimators=100,
            subsample=0.8,
            random_state=42
        ))
    ])

    # Hiperparametre grid'i
    if param_grid is None:
        param_grid = {'feature_selection__k': [5, 10, 15]}

        # GridSearchCV ile en iyi parametreleri bulma
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    print(f"En İyi Parametreler: {grid_search.best_params_}")
    print(f"Ortalama MSE: {np.mean(mse_scores):.4f}")
    print(f"Ortalama RMSE: {np.sqrt(np.mean(mse_scores)):.4f}")
    print(f"Ortalama R^2: {np.mean(r2_scores):.4f}")

    # Seçilen Özellikler
    selected_features = X.columns[best_model.named_steps['feature_selection'].get_support()]
    print("\nSeçilen Özellikler:")
    print(selected_features)

evaluate_xgb_with_feature_selection(X, y)