from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def linear_regression(xtrain, ytrain):
    lr = LinearRegression()
    lr.fit(xtrain, ytrain)
    return lr


def k_neighbours(xtrain, ytrain):
    knn = KNeighborsRegressor()
    knn.fit(xtrain, ytrain)
    return knn

