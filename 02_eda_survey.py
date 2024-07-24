import pandas as pd
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

# Adjusting Row Column Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###########
# READ DATA
###########

df_survey = pd.read_csv("./datasets/mental_survey_results.csv")

def check_df(dataframe, head=5):
    print("SHAPE".center(70,"-"))
    print(dataframe.shape)
    print("INFO".center(70,"-"))
    print(dataframe.info())
    print("MEMORY USAGE".center(70,"-"))
    print(f"{dataframe.memory_usage().sum() / (1024**2):.2f} MB")
    print("NUNIQUE".center(70,"-"))
    print(dataframe.nunique())
    print("MISSING VALUES".center(70,"-"))
    print(dataframe.isnull().sum())
    print("DUPLICATED VALUES".center(70,"-"))
    print(dataframe.duplicated().sum())
    
check_df(df_survey)


###########
# DROP & RENAME
###########

## Drop unnecessary columns
drop_list = ["Timestamp", "Foreign languages", "Composer", "Permissions"]

df_survey.drop(drop_list, axis=1, inplace=True)

## Reduction to 9 music genres
drop_genres = ["Frequency [Folk]", "Frequency [Gospel]", "Frequency [K pop]", "Frequency [Latin]", 
               "Frequency [Lofi]", "Frequency [Video game music]", "Frequency [Hip hop]"]

df_survey.drop(drop_genres, axis=1, inplace=True)

## Rename Cols
df_survey = df_survey.rename(columns={"Frequency [Classical]": "frequency_instrumental", "Frequency [Country]": "frequency_traditional",
                                      "Frequency [EDM]": "frequency_dance","Frequency [Pop]": "frequency_pop","Frequency [Jazz]": "frequency_jazz",
                                      "Frequency [Metal]": "frequency_metal","Frequency [R&B]": "frequency_rnb", "Frequency [Rap]": "frequency_rap",
                                      "Frequency [Rock]": "frequency_rock", "Primary streaming service": "streaming_service", "Hours per day": "hours_per_day",
                                      "While working": "while_working", "Fav genre": "fav_genre", "Music effects": "music_effects", "Age": "age", "Instrumentalist": "instrumentalist",
                                      "Exploratory": "exploratory", "BPM": "tempo", "Anxiety":"anxiety", "Depression":"depression", "Insomnia":"insomnia", "OCD": "obsession"})

## Rename Variables
tra_genres = ["Country", "Folk", "Gospel"]
df_survey.loc[df_survey["fav_genre"].isin(tra_genres), "fav_genre"] = "Traditional"
#############################################
jazz_genres = ["Jazz"]
df_survey.loc[df_survey["fav_genre"].isin(jazz_genres), "fav_genre"] = "Jazz"
#############################################
dance_genres = ["EDM", "Latin"]
df_survey.loc[df_survey["fav_genre"].isin(dance_genres), "fav_genre"] = "Dance"
#############################################
rb_genres = ["R&B"]
df_survey.loc[df_survey["fav_genre"].isin(rb_genres), "fav_genre"] = "R&B"
#############################################
rock_genres = ["Rock"]
df_survey.loc[df_survey["fav_genre"].isin(rock_genres), "fav_genre"] = "Rock"
#############################################
metal_genres = ["Metal"]
df_survey.loc[df_survey["fav_genre"].isin(metal_genres), "fav_genre"] = "Metal"
#############################################
pop_genres = ["Pop", "K pop"]
df_survey.loc[df_survey["fav_genre"].isin(pop_genres), "fav_genre"] = "Pop"
#############################################
inst_genres = ["Classical", "Video game music", "Lofi"]
df_survey.loc[df_survey["fav_genre"].isin(inst_genres), "fav_genre"] = "Instrumental"
#############################################
rap_genres = ["Hip hop", "Rap"]
df_survey.loc[df_survey["fav_genre"].isin(rap_genres), "fav_genre"] = "Rap"


###########
# DROP & RENAME
###########

col_list = ["frequency_instrumental", "frequency_traditional", "frequency_dance",
            "frequency_pop", "frequency_jazz", "frequency_metal",
            "frequency_rnb", "frequency_rap", "frequency_rock"]

df_survey[col_list] = df_survey[col_list].replace({"Very frequently":"Often"})
#############################################
df_survey["music_effects"] = df_survey["music_effects"].replace({"No effect":"No Effect"})
#############################################
replaced_streaming = {"Pandora":"Other",
                      "I do not use a streaming service.":"Other",
                      "Other streaming service":"Other"}
df_survey["streaming_service"] = df_survey["streaming_service"].replace(replaced_streaming)


###########
# MISSING VALUES & OUTLIERS
###########

## Fillna median for num cols
numeric_columns = df_survey.select_dtypes(include=["number"]).drop(columns=["tempo"]).columns
df_survey[numeric_columns] = df_survey[numeric_columns].fillna(df_survey[numeric_columns].median())

## Fillna mode for cat cols
categorical_columns = df_survey.select_dtypes(include=["object", "category"]).columns
df_survey[categorical_columns] = df_survey[categorical_columns].fillna(df_survey[categorical_columns].mode().iloc[0])

############################################

def impute_tempo(df):
    test_df = df[df["tempo"].isna()]

    if not test_df.empty:
        df_encoded = pd.get_dummies(df, drop_first=True)

        train_df_encoded = df_encoded.dropna(subset=["tempo"])
        test_df_encoded = df_encoded[df_encoded["tempo"].isna()]

        X_train = train_df_encoded.drop(columns=["tempo"])
        y_train = train_df_encoded["tempo"]
        X_test = test_df_encoded.drop(columns=["tempo"])

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        predicted_tempo = regressor.predict(X_test)

        df.loc[df["tempo"].isna(), "tempo"] = predicted_tempo

    return df

df_survey = impute_tempo(df_survey)

############################################

percentiles = [0.10, 0.25, 0.30, 0.40, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
df_survey.describe(percentiles=percentiles).T.style.background_gradient(axis=1, cmap="Reds")

############################################

# Function to calculate lower and upper thresholds
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Function to check for outliers in a specific column
def check_outlier(dataframe, col_name):
    if pd.api.types.is_numeric_dtype(dataframe[col_name]):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)][col_name]
    else:
        return pd.Series([])

# Function to replace outliers with defined thresholds
def replace_with_thresholds(dataframe, variable, q1=0.1, q3=0.9):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Selecting all numeric columns
numeric_vars = df_survey.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Iterating through each numeric column to check and handle outliers
for col in numeric_vars:
    outliers = check_outlier(df_survey, col)
    if not outliers.empty:
        print(f"Outliers found in {col}. Handling outliers...")
        replace_with_thresholds(df_survey, col)

print("Outlier handling completed.")

## Hours Per Day min 0.25
df_survey["hours_per_day"] = df_survey["hours_per_day"].clip(lower=0.25)
df_survey["hours_per_day"] = df_survey["hours_per_day"].replace({0.7:1})

## Tempo min 40 // max 250
df_survey["tempo"] = df_survey["tempo"].clip(lower=40, upper=250)


###########
# ASTYPE & QCUT
###########

df_survey["anxiety"] = df_survey["anxiety"].replace({7.5:7})
df_survey["anxiety"] = pd.qcut(df_survey["anxiety"], q=2, labels=[0, 1])
df_survey["anxiety"] = df_survey["anxiety"].astype(int)

df_survey["depression"] = df_survey["depression"].replace({3.5:3})
df_survey["depression"] = pd.qcut(df_survey["depression"], q=2, labels=[0, 1])
df_survey["depression"] = df_survey["depression"].astype(int) 

df_survey["insomnia"] = df_survey["insomnia"].replace({3.5:3})
df_survey["insomnia"] = pd.qcut(df_survey["insomnia"], q=2, labels=[0, 1])
df_survey["insomnia"] = df_survey["insomnia"].astype(int) 

df_survey["obsession"] = df_survey["obsession"].replace({8.5:9, 5.5:6}) 
df_survey["obsession"] = pd.qcut(df_survey["obsession"], q=2, labels=[0, 1])
df_survey["obsession"] = df_survey["obsession"].astype(int) 

df_survey["age"] = df_survey["age"].astype(int)

df_survey.info()


###########
# EXPORT DATA
###########

df_survey.to_csv("./datasets/mental_final.csv", index=False)