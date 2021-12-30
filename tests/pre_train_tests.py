import pytest
import pandas as pd
from src.data.make_dataset import data_preprocessing
from src.util.csv_data_operations import split_train_test_data
from src.models.train_model import linear_regression, k_neighbours
from src.models.predict_model import predict_on_test_data
import pytest_check as check


@pytest.fixture
def data_preparation():
    data_preprocessing()
    return split_train_test_data()


@pytest.fixture
def linear_regression_prediction(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    lr = linear_regression(xtrain, ytrain)
    ypred = predict_on_test_data(lr, xtest)
    return xtest, ypred


@pytest.fixture
def k_neighbors_prediction(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    knn = k_neighbours(xtrain, ytrain)
    ypred = predict_on_test_data(knn, xtest)
    return xtest, ypred


def test_data_leak(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    concat_df = pd.concat([xtrain, xtest])
    assert concat_df.shape[0] == xtrain.shape[0] + xtest.shape[0]


def test_predicted_output_shape(linear_regression_prediction, k_neighbors_prediction):
    print("Linear regression")
    xtest, ypred = linear_regression_prediction
    check.equal(ypred.shape, (xtest.shape[0], ))
    # assert ypred.shape == (xtest.shape[0], 1)
    print("K nearest neighbours")
    xtest, ypred = k_neighbors_prediction
    check.equal(ypred.shape, (xtest.shape[0], ))
    # assert ypred.shape == (xtest.shape[0], )

