import pandas as pd
from sklearn.model_selection import train_test_split


def load_data_from_path(filepath):
    df = pd.read_csv(filepath)
    return df


def load_insurance_data():
    input_path = "data/raw/insurance.csv"
    df = load_data_from_path(input_path)
    return df


def save_insurance_data(df: pd.DataFrame):
    output_path = "data/processed/processed_insurance_data.csv"
    df.to_csv(output_path)
    return


def load_processed_insurance_data():
    input_path = "data/processed/processed_insurance_data.csv"
    df = load_data_from_path(input_path)
    return df


def save_train_test_data(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, Xtest: pd.DataFrame, Ytest: pd.DataFrame):
    Xtrain.to_csv("data/model_training_data/training_data.csv")
    Ytrain.to_csv("data/model_training_data/training_data_result.csv")
    Xtest.to_csv("data/model_testing_data/testing_data.csv")
    Ytest.to_csv("data/model_testing_data/testing_data_result.csv")
    Ytest.to_csv("data/model_testing_data/testing_data_result.csv")
    return


def split_train_test_data():
    df = load_processed_insurance_data()
    X = df.drop(columns=["charges"])
    # In the above line, the column sex is also dropped but let's see what's the effect of keeping sex on the
    # invariance test
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    save_train_test_data(X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test
