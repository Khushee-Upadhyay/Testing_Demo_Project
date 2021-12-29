import pandas as pd
import pytest
from src.data.make_dataset import data_preprocessing
from src.util.csv_data_operations import load_processed_insurance_data, split_train_test_data


@pytest.fixture
def data_preparation():
    data_preprocessing()
    return split_train_test_data()


def test_data_leak(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    concat_df = pd.concat([xtrain, xtest])
    assert concat_df.shape[0] == xtrain.shape[0] + xtest.shape[0]


def test_predicted_output_shape(xtest, pred_train):
    assert pred_train.shape == (xtest.shape[0],)
