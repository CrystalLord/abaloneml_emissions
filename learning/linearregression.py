from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
import numpy as np

def regr(X,y):
    """Returns the coefficient of the linear regression."""
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

def dummy(X, y):
    """Returns the dummy regressor, trained on the provided data."""
    dummy = DummyRegressor('mean')
    dummy.fit(X, y)
    return dummy
