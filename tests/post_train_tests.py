import numpy as np
import pytest
import pytest_check as check
from src.data.make_dataset import data_preprocessing
from src.models.train_model import linear_regression, k_neighbours

from src.util.csv_data_operations import split_train_test_data


@pytest.fixture
def data_preparation():
    data_preprocessing()
    return split_train_test_data()


@pytest.fixture
def return_models(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    lr = linear_regression(xtrain, ytrain)
    knn = k_neighbours(xtrain, ytrain)
    return [lr, knn]


def test_sex_invariance(return_models):
    models = return_models
    for model in models:
        print("Checking for " + str(model.__class__.__name__))
        female_sample = [19, 1, 27.9, 0, 1, 2, 1, 1]
        male_sample = [19, 0, 27.9, 0, 1, 2, 1, 1]
        result_female_sample = model.predict(np.array(female_sample).reshape(1, -1))
        result_male_sample = model.predict(np.array(male_sample).reshape(1, -1))
        assert result_female_sample == result_male_sample
