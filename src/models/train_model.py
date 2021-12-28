from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


def linear_regression(xtrain, ytrain):
    lr = LinearRegression()
    lr.fit(xtrain, ytrain)
    return lr


def k_neighbours(xtrain, ytrain):
    knn = KNeighborsRegressor()
    knn.fit(xtrain, ytrain)
    return knn


def decision_tree(xtrain, ytrain):
    tree = DecisionTreeRegressor()
    tree.fit(xtrain, ytrain)
    return decision_tree
