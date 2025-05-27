import numpy as np
from typing import Callable, NoReturn
from base_estimator import BaseEstimator
from loss_functions import misclassification_error

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array")

        # Initialize the weights for each sample
        self.D_ = []
        self.D_.append(np.ones(X.shape[0]) / X.shape[0])
        self.models_ = []
        self.weights_ = []

        for t in range(self.iterations_):
            
            # Fit a weak learner
            idx = np.random.choice(X.shape[0], size=X.shape[0], p=self.D_[t])
            sampled_X, sampled_y = X[idx], y[idx]

            model = self.wl_()
            model._fit(sampled_X, sampled_y)
            pred = model._predict(X)

            # Calculate the error
            err = np.sum(self.D_[t][y != pred]) / np.sum(self.D_[t])

            # Calculate the weight of the weak learner
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

            # Append the model and its weight
            self.models_.append(model)
            self.weights_.append(alpha)

            # Update the weights for each sample
            if t == self.iterations_ - 1:
                # If this is the last iteration, we do not need to update D_
                break

            self.D_.append(self.D_[t] * np.exp(-alpha * y * pred))
            self.D_[t+1] /= np.sum(self.D_[t+1])  # Normalize the weights

        return None

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred = np.zeros(X.shape[0], dtype=float)
        for t in range(self.iterations_):
            pred += self.models_[t]._predict(X) * self.weights_[t]
        return np.sign(pred)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X), normalize=True)
        

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred = np.zeros(X.shape[0], dtype=float)
        for t in range(T):
            pred += self.models_[t]._predict(X) * self.weights_[t] # TODO - check predict/_predict
        return np.sign(pred)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        pred = self.partial_predict(X, T)
        return misclassification_error(y, pred, normalize=True)