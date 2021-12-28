import pandas as pd


def test_predicted_output_shape(xtest, pred_train):
    assert pred_train.shape == (xtest.shape[0],)


def test_data_leak(xtrain, xtest):
    concat_df = pd.concat([xtrain, xtest])
    assert concat_df.shape[0] ==  xtrain.shape[0] + xtest.shape[0]

