import pandas as pd
import numpy as np

def noise_processing_func(df):
    df_copy = df.copy()
    df_copy["Fuel_station_manager"] = df_copy["Fuel_station_manager"].where(
    df_copy["Fuel_station_manager"].map(df_copy["Fuel_station_manager"].value_counts()) >= 20, "Other")
    df_copy["City"] = df_copy["City"].where(df_copy["City"].map(df_copy["City"].value_counts()) >= 20, "Other") 
    return df_copy

def datetime_processing_func(df_full_eda, date_feature):
    df_full_eda["DateTime"] = pd.to_datetime(df_full_eda[date_feature])
    df_full_eda["Date"] = df_full_eda["DateTime"].apply(lambda x: x.date())
    df_full_eda["Date"] = pd.to_datetime(df_full_eda["Date"])
    df_full_eda["Date"] = df_full_eda["Date"].dt.dayofyear
    df_full_eda["Time"] = df_full_eda["DateTime"].apply(lambda x: x.time())
    df_full_eda["Time"] = df_full_eda["Time"].apply(lambda x: x.hour*60 + x.minute)
    df_full_eda = df_full_eda.drop(date_feature, axis=1)
    df_full_eda = df_full_eda.drop(["DateTime"], axis=1)
    return df_full_eda

def train_test_splitting(df):
    df = df.drop_duplicates().sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_learning, df_check = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = df_learning.drop(columns="Price"), df_learning["Price"]
    X_test,  y_test  = df_check.drop(columns="Price"),    df_check["Price"]
    return X_train, X_test, y_train, y_test
    
def to_categorical_func(X, cols, categories=None):
    X = X.copy()
    for c in cols:
        if categories is None:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.Categorical(X[c], categories=categories[c])
    return X

def fill_unknown(X, cols, token="missing"):
    X = X.copy()
    for c in cols:
        X[c] = X[c].cat.add_categories(token).fillna(token)
    return X