from sklearn import linear_model
import numpy as np

def regr(X,y):
    """Returns the coefficient of the linear regression."""
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    return reg

def ridge(X,y, alpha=0.5):
	reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X, y)
    return reg

def lasso(X,y, alpha=0.5):
	reg = linear_model.Lasso(alpha=alpha)
    reg.fit(X, y)
    return reg