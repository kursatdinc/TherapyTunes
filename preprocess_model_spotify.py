import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA
###########

df_spoti_model = pd.read_csv("./datasets/spotify_model.csv")

###########
# FEATURE ENGINEERING PIPELINE
###########

scaled_cols = ["tempo", "valence", "energy"]
pass_cols = ["anxiety_index", "depression_index", "insomnia_index"]

preprocessor = ColumnTransformer(
    transformers=[
        ("pass_cols", "passthrough", pass_cols),
        ("sc_cols", StandardScaler(), scaled_cols)   
    ]
)

preprocessing_pipeline = Pipeline([("preprocessor", preprocessor)])

###########
joblib.dump(preprocessing_pipeline, "./models/spotify_preprocessing.pkl")
###########

def preprocess_df(new_data, pipeline):
    
    preprocessed_data = pipeline.named_steps["preprocessor"].transform(new_data)
    
    feature_names = (
        pipeline.named_steps["preprocessor"].named_transformers_["pass_cols"].get_feature_names_out().tolist() +
        pipeline.named_steps["preprocessor"].named_transformers_["sc_cols"].get_feature_names_out().tolist()
    )

    preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
    
    return preprocessed_df

new_user = df_spoti_model.sample(1)

preprocessing_pipeline.fit(df_spoti_model)

preprocessed_data_X = preprocess_df(df_spoti_model, preprocessing_pipeline)


###########
# MODEL
###########

y = df_spoti_model["cluster"]
X = preprocessed_data_X

lgbm_params = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "n_estimators": 500,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
        }
    
lgbm_model = LGBMClassifier(**lgbm_params, verbose=-1).fit(X, y)

random_user = X.sample(1)

predicted_cluster = lgbm_model.predict(random_user)[0]

joblib.dump(lgbm_model, "./models/spotify_model.pkl")