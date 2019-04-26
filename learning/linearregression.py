from sklearn import linear_model
import numpy as np

def regr():
    """Returns the coefficient of the linear regression."""
    reg = linear_model.LinearRegression()
    return reg

def ridge(alpha=1.0):
    reg = linear_model.Ridge(alpha=alpha)
    return reg

def lasso(alpha=1.0):
    reg = linear_model.Lasso(alpha=alpha)
    return reg