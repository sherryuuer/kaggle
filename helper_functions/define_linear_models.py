import pandas as pd
from sklearn import linear_model
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
        ('Ridge', linear_model.Ridge()),
        ('Lasso', linear_model.Lasso()),
        ('ElasticNet', linear_model.ElasticNet()),
        ('SGDRegressor', linear_model.SGDRegressor())
        # Add more linear models if needed
    ]

    # Train each linear model
    for name, model in linear_models:
        model.fit(X_train, y_train)
        models[name] = model

    return models
