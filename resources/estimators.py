from abc import ABC, abstractmethod
from typing import NoReturn
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Lasso as SklearnLasso
import numpy as np
from base_estimator import BaseEstimator


class LinearRegression(BaseEstimator):
    def __init__(self):
        self.model = SklearnLR()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        predictions = self._predict(X)
        return np.mean((predictions - y) ** 2)


class Lasso(BaseEstimator):
    def __init__(self, alpha: float = 1.0, include_intercept: bool = True):
        self.include_intercept_ = include_intercept
        self.model = SklearnLasso(alpha=alpha, fit_intercept=include_intercept, max_iter=10000)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        predictions = self._predict(X)
        return np.mean((predictions - y) ** 2)


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    notes:
        Implement the RidgeRegression class in estimators.py as specified in the class documentation. You are not allowed to use the scikit-learn implementation of Ridge Regression.
        • Note that in RidgeRegression, you should provide an option to include an intercept (a
        boolean argument in the constructor), but the regularization should not be applied to the
        intercept term
    """

    def __init__(self, lam: float, include_intercept: bool = True):
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        n_samples, n_features = X.shape
        if self.include_intercept_:
            X = np.hstack((np.ones((n_samples, 1)), X))
            n_features += 1

        # Compute the closed-form solution for Ridge Regression
        I = np.eye(n_features)
        if self.include_intercept_: # not considering regularization for intercept term
            I[0, 0] = 0
        XTX = X.T @ X + self.lam_ * I
        XTy = X.T @ y
        self.coefs_ = np.linalg.inv(XTX) @ XTy

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        predictions = self._predict(X)
        return np.mean((predictions - y) ** 2)


