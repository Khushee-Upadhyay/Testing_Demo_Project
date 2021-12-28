import pandas as pd


def predict(model, xtest):
    y_test = model.predict(xtest)
    filename = str(model.__class__.__name__)+"predicted output.csv"
    prediction = pd.DataFrame(y_test).to_csv("data/model_testing_data/"+filename)
    return
