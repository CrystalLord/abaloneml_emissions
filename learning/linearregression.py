from sklearn.linear_model import LinearRegression
import numpy as np

def regr(X,y):
    """Returns the coefficient of the linear regression."""
    reg = LinearRegression()
    reg.fit(X, y)
    return reg
