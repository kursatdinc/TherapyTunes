import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA
###########

df_survey = pd.read_csv("./datasets/mental_final.csv")

###########
# FEATURE ENGINEERING PIPELINE
###########

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

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
        
        X_["age_group"] = X_["age"].apply(get_age_group)
        X_["age_group"] = X_["age_group"].astype(int)

        # Feature 2
        freq_cols = [col for col in X_.columns if "frequency" in col]

        ordinal_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

        for col in freq_cols:
            X_[col] = X_[col].map(ordinal_mapping)

        X_["average_frequency"] = X_[freq_cols].mean(axis=1)

        # Feature 3
        def calculate_genre_diversity(row):
            non_zero_genres = sum(1 for value in row if value > 0)
            return non_zero_genres / len(freq_cols)
        X_["genre_diversity"] = X_[freq_cols].apply(calculate_genre_diversity, axis=1)

        # Feature 4
        X_["normalized_hours"] = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_[["hours_per_day"]])
        X_["normalized_diversity"] = X_["genre_diversity"]
        X_["normalized_frequency"] = X_["average_frequency"] / 3

        X_["music_consumption_profile"] = (X_["normalized_hours"] * 0.3 +
                                                  X_["normalized_diversity"] * 0.3 +
                                                  X_["normalized_frequency"] * 0.4)

        drop = ["normalized_hours", "normalized_diversity", "normalized_frequency"]
        X_.drop(columns=drop, axis=1, inplace=True)

        # Feature 5
        X_["rock_metal_affinity"] = (X_["frequency_metal"] + X_["frequency_rock"] + 1) / 2
        
        # Feature 6
        X_["mainstream_music_score"] = X_["average_frequency"] * (1 - X_["genre_diversity"])

        # Drop original columns that are no longer needed
        X_ = X_.drop(columns="average_frequency")

        return X_


numeric_features = ["age", "age_group", "hours_per_day", "genre_diversity", "music_consumption_profile",
                    "rock_metal_affinity", "mainstream_music_score"]

binary_features = ["while_working", "instrumentalist", "exploratory"]

categorical_features = ["streaming_service", "fav_genre"]

frequency_features = ["frequency_instrumental", "frequency_traditional", "frequency_dance",
                      "frequency_jazz", "frequency_metal", "frequency_pop", "frequency_rnb",
                      "frequency_rap", "frequency_rock"]

musiceffect_feature = ["music_effects"]


preprocessor = ColumnTransformer(
    transformers=[
        ("bin", OneHotEncoder(drop="first", sparse_output=False), binary_features),
        ("freq", StandardScaler(), frequency_features),
        ("musiceffect", OneHotEncoder(drop="first", sparse_output=False), musiceffect_feature),
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
    ])

preprocessing_pipeline = Pipeline([("feature_engineer", FeatureEngineer()),
                                   ("preprocessor", preprocessor)])


###########
joblib.dump(preprocessing_pipeline, "./models/survey_preprocessing.pkl")
###########


def preprocess_df(new_data, pipeline):

    fe_data = pipeline.named_steps["feature_engineer"].transform(new_data)
    
    preprocessed_data = pipeline.named_steps["preprocessor"].transform(fe_data)
    
    feature_names = (
        pipeline.named_steps["preprocessor"].named_transformers_["bin"].get_feature_names_out().tolist() +
        pipeline.named_steps["preprocessor"].named_transformers_["freq"].get_feature_names_out().tolist() +
        pipeline.named_steps["preprocessor"].named_transformers_["musiceffect"].get_feature_names_out().tolist() +
        pipeline.named_steps["preprocessor"].named_transformers_["num"].get_feature_names_out().tolist() +
        pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out().tolist()
    )

    preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
    
    return preprocessed_df

new_user = df_survey.sample(1)

preprocessing_pipeline.fit(df_survey)

preprocessed_data_X = preprocess_df(df_survey, preprocessing_pipeline)


###########
# MODELS
###########

###################### TEMPO ######################

y = df_survey["tempo"]
X = preprocessed_data_X

xgboost_params = {"learning_rate":0.01,
                  "max_depth":3,
                  "n_estimators":100,
                  "subsample":0.8,
                  "random_state":42}

xgboost_model = XGBRegressor(**xgboost_params).fit(X, y)

random_user = X.sample(1)

predicted_tempo = xgboost_model.predict(random_user)[0]

joblib.dump(xgboost_model, "./models/tempo_model.pkl")

#RMSE: 31.80


###################### ANXIETY ######################

y = df_survey["anxiety"]
X = preprocessed_data_X

abc_params = {"learning_rate": 0.1,
              "n_estimators": 100}

abc_model = AdaBoostClassifier(**abc_params, random_state=42).fit(X, y)

random_user = X.sample(1)

predicted_anxiety = abc_model.predict_proba(random_user)[0][1]

joblib.dump(abc_model, "./models/anx_model.pkl")

#Accuracy: 0.57
#F1 Score: 0.58
#Recall: 0.61
#Precision: 0.55
#ROC AUC: 0.59


###################### DEPRESSION ######################

y = df_survey["depression"]
X = preprocessed_data_X

svc_params = {"C": 1,
              "kernel": "rbf",
              "degree": 3,
              "coef0": 0.0}


svc_model = SVC(**svc_params, probability=True, random_state=42).fit(X, y)

random_user = X.sample(1)

predicted_depression = svc_model.predict_proba(random_user)[0][1]

joblib.dump(svc_model, "./models/dep_model.pkl")

#Accuracy: 0.61
#F1 Score: 0.56
#Recall: 0.53
#Precision: 0.60
#ROC AUC: 0.65


###################### INSOMNIA ######################

y = df_survey["insomnia"]
X = preprocessed_data_X

rfc_params = {"criterion": "gini",
              "max_depth": None,
              "n_estimators": 100}

rfc_model = RandomForestClassifier(**rfc_params, random_state=42).fit(X, y)

random_user = X.sample(1)

predicted_insomnia = rfc_model.predict_proba(random_user)[0][1]

joblib.dump(rfc_model, "./models/ins_model.pkl")

#Accuracy: 0.58
#F1 Score: 0.53
#Recall: 0.50
#Precision: 0.56
#ROC AUC: 0.63