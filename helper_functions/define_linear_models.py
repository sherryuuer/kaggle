import pandas as pd
from sklearn import linear_model, svm, neighbors, tree
from sklearn.metrics import mean_squared_error


def define_linear_model(X_train, y_train):
    """
    Define and train linear models available in scikit-learn.

    Parameters:
    X_train: Feature data for training.
    y_train: Target data for training.

    Returns:
    models: A dictionary containing trained linear models.
    """

    # Initialize a dictionary to store trained models
    models = {}

    # List of linear models available in scikit-learn
    linear_models = [
        ('LinearRegression', linear_model.LinearRegression()),
        ('Ridge', linear_model.Ridge(alpha=0.5)),
        ('Lasso', linear_model.Lasso(alpha=0.5)),
        ('ElasticNet', linear_model.ElasticNet(l1_ratio=0.5)),
        ('SGDRegressor', linear_model.SGDRegressor()),
        ('SVR', svm.SVR(kernel="linear")),
        ('KNeighbors', neighbors.KNeighborsRegressor(n_neighbors=5)),
        ('DTRegressor', tree.DecisionTreeRegressor())
        ('BayesianRidge', linear_model.BayesianRidge())
        # Add more linear models if needed
    ]

    # Train each linear model
    for name, model in linear_models:
        model.fit(X_train, y_train)
        models[name] = model

    return models
