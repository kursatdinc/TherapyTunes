import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###########
mental_final_df = pd.read_csv("./datasets/mental_final.csv")
mental_final_df.head(10)
###########

###########
spotify_final_df = pd.read_csv("./datasets/spotify_final.csv")
spotify_final_df.head(10)
###########

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Duplicated Values #####################")
    print(dataframe.duplicated().sum())
    print("##################### Missing Values #####################")
    print(dataframe.isnull().sum())
    print("##################### Number of Unique Values #####################")
    print(dataframe.nunique())

check_df(mental_final_df)

# tempo kontrol edilecek.
# hours per day kontrol edilecek.
# hepsinin outlier değerleri kontrol edilecek.

mental_final_df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

mental_final_df.info()

mental_final_df["anxiety"] = mental_final_df["anxiety"].astype(object)
mental_final_df["depression"] = mental_final_df["depression"].astype(object)
mental_final_df["insomnia"] = mental_final_df["insomnia"].astype(object)


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

cat_cols, num_cols, cat_but_car = grab_col_names(mental_final_df)


def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].shape[0] > 0:
        return True
    else:
        return False
    
for col in num_cols:
    print(col, check_outlier(mental_final_df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(mental_final_df, col)

mental_final_df["hours_per_day"].sort_values(ascending=False)
mental_final_df["age"].sort_values(ascending=False)
mental_final_df["tempo"].sort_values(ascending=False)

mental_final_df.info()


###########################
mental_final_df.to_csv("./datasets/mental_after_eda.csv", index=False)
###########################