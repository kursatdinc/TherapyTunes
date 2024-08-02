import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier
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

        # Feature 1: Generation
        def get_generation(age):
            if age >= 77:
                return "Silent Generation"
            elif age >= 59:
                return "Baby Boomer"
            elif age >= 43:
                return "Generation X"
            elif age >= 27:
                return "Millennial"
            elif age >= 11:
                return "Generation Z"
            else:
                return "Generation Alpha"
        
        X_["generation"] = X_["age"].apply(get_generation)

        # Feature 2: Listening Habit
        X_["listening_habit"] = pd.cut(X_["hours_per_day"], 
                                       bins=[0, 2, 4, 7, 24],
                                       labels=[0, 1, 2, 3],
                                       include_lowest=True)
        X_["listening_habit"] = X_["listening_habit"].astype(int)

        # Feature 3: Average Frequency
        freq_cols = [col for col in X_.columns if "frequency" in col]
        ordinal_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
        for col in freq_cols:
            X_[col] = X_[col].map(ordinal_mapping)
        X_["average_frequency"] = X_[freq_cols].mean(axis=1)

        # Feature 4: Genre Diversity
        def calculate_genre_diversity(row):
            non_zero_genres = sum(1 for value in row if value > 0)
            return non_zero_genres / len(freq_cols)
        X_["genre_diversity"] = X_[freq_cols].apply(calculate_genre_diversity, axis=1)

        # Feature 5: Daily Listening Intensity
        X_["daily_listening_intensity"] = (X_["hours_per_day"] * X_["listening_habit"]) / 100

        # Feature 6: Rock Metal Affinity
        X_["rock_metal_affinity"] = (X_["frequency_metal"] + X_["frequency_rock"]) / 2

        # Feature 7: Genre Diversity Ratio
        X_["genre_diversity_ratio"] = X_.apply(lambda row: row["genre_diversity"] / row["average_frequency"] 
                                               if row["average_frequency"] != 0 else 0, axis=1)
        
        # Feature 8: Mainstream Music Score
        X_["mainstream_music_score"] = X_["average_frequency"] * (1 - X_["genre_diversity"])

        # Feature 9: Frequency Diversity Ratio
        X_["frequency_diversity_ratio"] = X_.apply(lambda row: row["average_frequency"] / row["genre_diversity"]
                                                   if row["genre_diversity"] != 0 else 0, axis=1)
        
        # Feature 10: Urban Frequency Interaction
        X_["urban_frequency_interaction"] = ((X_["frequency_rnb"] + X_["frequency_rap"]) / 2) * X_["average_frequency"]

        # Feature 11: Metal Music Exposure
        X_["metal_music_exposure"] = X_["hours_per_day"] * X_["frequency_metal"]

        # Drop original columns that are no longer needed
        X_ = X_.drop(columns=["hours_per_day"])

        return X_


numeric_features = ["age", "listening_habit", "average_frequency", "genre_diversity", 
                    "daily_listening_intensity", "rock_metal_affinity", "genre_diversity_ratio",
                    "mainstream_music_score", "frequency_diversity_ratio", 
                    "urban_frequency_interaction", "metal_music_exposure"]

binary_features = ["while_working", "instrumentalist", "exploratory"]

categorical_features = ["streaming_service", "fav_genre", "generation"]

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
    
    feature_names = (binary_features + frequency_features + musiceffect_feature + numeric_features +
                     pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features).tolist())
    
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
    
    return preprocessed_df

preprocessing_pipeline.fit(df_survey)

preprocessed_data_X = preprocess_df(df_survey, preprocessing_pipeline)


###########
# MODELS
###########

###################### TEMPO ######################

y = df_survey["tempo"]
X = preprocessed_data_X

catboost_params = {"border_count": 64,
                   "depth": 3,
                   "iterations": 300,
                   "l2_leaf_reg": 5,
                   "learning_rate": 0.01}

catboost_model = CatBoostRegressor(**catboost_params, silent=True)

xgboost_params = {"colsample_bytree": 0.8,
                  "learning_rate": 0.01,
                  "max_depth": 3,
                  "n_estimators": 100,
                  "subsample": 0.7}

xgboost_model = XGBRegressor(**xgboost_params)

# Ensemble Modelling
ensemble_model = VotingRegressor(estimators=[("xgb", xgboost_model),
                                             ("catboost", catboost_model)]).fit(X, y)

random_user = X.sample(1)

predicted_tempo = ensemble_model.predict(random_user)[0]

joblib.dump(ensemble_model, "./models/tempo_model.pkl")


###################### ANXIETY ######################

y = df_survey["anxiety"]
X = preprocessed_data_X

cart_params = {"criterion": "entropy",
               "max_depth": 7,
               "min_samples_leaf": 2,
               "min_samples_split": 6}

cart_model = DecisionTreeClassifier(**cart_params, random_state=42).fit(X, y)

random_user = X.sample(1)

predicted_anxiety = cart_model.predict_proba(random_user)[0][1]

joblib.dump(cart_model, "./models/anx_model.pkl")


###################### DEPRESSION ######################

y = df_survey["depression"]
X = preprocessed_data_X

lgbm_params = {"colsample_bytree": 1.0,
               "learning_rate": 0.1,
               "max_depth": 5,
               "n_estimators": 100,
               "num_leaves": 31,
               "subsample": 0.8,
               "verbose": -1}

lgbm_model = LGBMClassifier(**lgbm_params, random_state=42).fit(X, y)

random_user = X.sample(1)

predicted_depression = lgbm_model.predict_proba(random_user)[0][1]

joblib.dump(lgbm_model, "./models/dep_model.pkl")


###################### INSOMNIA ######################

y = df_survey["insomnia"]
X = preprocessed_data_X

svc_params = {"C": 1,
              "kernel": "linear",  
              "degree": 3,
              "coef0": 0.0}

svc_model = SVC(**svc_params, random_state=42, probability=True).fit(X, y)

random_user = X.sample(1)

predicted_insomnia = svc_model.predict_proba(random_user)[0][1]

joblib.dump(svc_model, "./models/ins_model.pkl")