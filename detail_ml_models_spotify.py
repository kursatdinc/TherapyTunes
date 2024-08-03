import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA
###########

df_spoti_model = pd.read_csv("./datasets/spotify_model.csv")


###########
# FEATURE ENGINEERING
###########

###################### ENCODING & SCALING ######################

scaler = StandardScaler()

df_spoti_model[["tempo", "valence", "energy"]] = scaler.fit_transform(df_spoti_model[["tempo", "valence", "energy"]])


###########
# MODELS
###########


##########
# LGBM
##########

def evaluate_lgbm_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    params = {
        'objective': 'multiclass',
        'num_class': 5,  # Sınıf sayınız
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'n_estimators': 500,
        'max_depth': 7,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8
        }
    
    model = LGBMClassifier(**params, random_state=random_state, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance = evaluate_lgbm_model(df_spoti_model, "cluster")

for metric, value in performance.items():
    print(f"{metric}: {value:.4f}")

# Accuracy: 0.7672
# F1 Score: 0.7549
# Precision: 0.7570
# Recall: 0.7672



def evaluate_lgbm_model_with_cv(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 63],
        'max_depth': [-1, 10, 20]
    }

    model = LGBMClassifier(random_state=random_state, verbose=-1)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Best Parameters": best_params
    }

    return performance_metrics
performance = evaluate_lgbm_model_with_cv(df_spoti_model, "cluster")
for metric, value in performance.items():
    print(f"{metric}: {value}")




##########
# XGB
##########

def evaluate_xgb_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Eğitim ve test setlerine ayırın
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # XGBClassifier ile model oluşturun
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_xgb = evaluate_xgb_model(df_spoti_model, "cluster")

for metric, value in performance_xgb.items():
    print(f"{metric}: {value:.4f}")

#Accuracy: 0.35
#F1 Score: 0.27
#Precision: 0.31
#Recall: 0.35



def evaluate_xgb_model_with_hyperopt(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # XGBClassifier ile model oluşturun
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # RandomizedSearchCV kullanarak hiperparametre optimizasyonu yapın
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=3, random_state=random_state, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # En iyi modeli alın
    best_model = random_search.best_estimator_

    # Test seti ile tahmin yapın
    y_pred = best_model.predict(X_test)

    # Model performansını değerlendirin
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Performans metriklerini içeren bir sözlük döndürün
    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_xgb = evaluate_xgb_model_with_hyperopt(df_spoti_model, "cluster")

for metric, value in performance_xgb.items():
    print(f"{metric}: {value:.4f}")




##########
# RFC
##########

def evaluate_rfc_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestClassifier(random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_rfc = evaluate_rfc_model(df_spoti_model, "cluster")

for metric, value in performance_rfc.items():
    print(f"{metric}: {value:.4f}")

#Accuracy: 0.4207
#F1 Score: 0.3915
#Precision: 0.3885
#Recall: 0.4207



def evaluate_rfc_model_with_hyperopt(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Eğitim ve test setlerine ayırın
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # RandomForestClassifier ile model oluşturun
    model = RandomForestClassifier(random_state=random_state)

    # Hiperparametre aralığını belirleyin
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # RandomizedSearchCV kullanarak hiperparametre optimizasyonu yapın
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=3, random_state=random_state, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # En iyi modeli alın
    best_model = random_search.best_estimator_

    # Test seti ile tahmin yapın
    y_pred = best_model.predict(X_test)

    # Model performansını değerlendirin
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Performans metriklerini içeren bir sözlük döndürün
    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_rfc = evaluate_rfc_model_with_hyperopt(df_spoti_model, "cluster")

for metric, value in performance_rfc.items():
    print(f"{metric}: {value:.4f}")





##########
# DTC
##########


def evaluate_dtc_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = DecisionTreeClassifier(random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_dtc = evaluate_dtc_model(df_spoti_model, "cluster")

for metric, value in performance_dtc.items():
    print(f"{metric}: {value:.4f}")

#Accuracy: 0.27
#F1 Score: 0.27
#Precision: 0.27
#Recall: 0.27



def evaluate_dtc_model_with_hyperopt(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_state),
                               param_grid=param_grid,
                               cv=5,
                               scoring='f1_weighted',
                               n_jobs=-1,
                               verbose=1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Best Parameters": best_params
    }

    return performance_metrics

performance_dtc = evaluate_dtc_model_with_hyperopt(df_spoti_model, "cluster")

for metric, value in performance_dtc.items():
    print(f"{metric}: {value}")




##########
# SVC
##########

def evaluate_svc_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = SVC(probability=True, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_svc = evaluate_svc_model(df_spoti_model, "cluster")

for metric, value in performance_svc.items():
    print(f"{metric}: {value:.4f}")




def evaluate_svc_model_with_hyperopt(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(SVC(probability=True, random_state=random_state), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_svc = evaluate_svc_model_with_hyperopt(df_spoti_model, "cluster")

for metric, value in performance_svc.items():
    print(f"{metric}: {value:.4f}")




##########
# LR
##########

def evaluate_lr_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LogisticRegression(max_iter=1000, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_lr = evaluate_lr_model(df_spoti_model, "cluster")

for metric, value in performance_lr.items():
    print(f"{metric}: {value:.4f}")

#Accuracy: 0.4207
#F1 Score: 0.3915
#Precision: 0.3885
#Recall: 0.4207




def evaluate_lr_model_with_hyperopt(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_lr = evaluate_lr_model(df_spoti_model, "cluster")

for metric, value in performance_lr.items():
    print(f"{metric}: {value:.4f}")




##########
# GBC
##########

def evaluate_gbc_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = GradientBoostingClassifier(random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_gbc = evaluate_gbc_model(df_spoti_model, "cluster")

for metric, value in performance_gbc.items():
    print(f"{metric}: {value:.4f}")



def evaluate_gbc_model_with_hyperopt(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    model = GradientBoostingClassifier(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_gbc = evaluate_gbc_model(df_spoti_model, "cluster")

for metric, value in performance_gbc.items():
    print(f"{metric}: {value:.4f}")




##########
# KNN
##########

def evaluate_knn_model(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_knn = evaluate_knn_model(df_spoti_model, "cluster")

for metric, value in performance_knn.items():
    print(f"{metric}: {value:.4f}")




def evaluate_knn_model_with_hyperopt(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

    return performance_metrics

performance_knn = evaluate_knn_model(df_spoti_model, "cluster")

for metric, value in performance_knn.items():
    print(f"{metric}: {value:.4f}")