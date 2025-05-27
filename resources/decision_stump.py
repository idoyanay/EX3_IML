from __future__ import annotations
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

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

        # Initialize the best threshold and error
        best_thr, best_thr_err = None, None
        for feature in range(X.shape[1]):
            # for this feature, find the best threshold
            for sign in [-1, 1]:
                # find the threshold and error
                thr, thr_err = self._find_threshold(X[:, feature], y, sign)
                # if this is the best threshold so far, save it
                if best_thr is None or thr_err < best_thr_err:
                    best_thr, best_thr_err = thr, thr_err
                    self.j_, self.sign_ = feature, sign
        self.threshold_ = best_thr

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        if self.j_ is None or self.threshold_ is None or self.sign_ is None:
            raise ValueError("The model has not been fitted yet. Call fit() before predict().")
        predictions = np.zeros(X.shape[0], dtype=int)
        # Get the feature values
        predictions[X[:, self.j_] < self.threshold_] = -self.sign_
        predictions[X[:, self.j_] >= self.threshold_] = self.sign_
        return predictions

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Sort the values and labels together
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]
        # insert to the sorted values min and max values to ensure that the threshold can be set to these values
        sorted_values = np.insert(sorted_values, 0, sorted_values[0] - 1)
        sorted_values = np.append(sorted_values, sorted_values[-1] + 1)

        # Initialize the best threshold and error
        best_thr, best_thr_err = None, np.inf

        # Iterate over all possible thresholds
        for i in range(len(sorted_values) - 1):

            # Calculate the threshold as the average of two consecutive values
            thr = (sorted_values[i] + sorted_values[i + 1]) / 2

            # Calculate the predictions based on the threshold
            predictions = np.where(values < thr, -sign, sign)

            # Calculate the misclassification error
            thr_err = misclassification_error(labels, predictions)

            # Update the best threshold if necessary
            if best_thr is None or thr_err < best_thr_err:
                best_thr, best_thr_err = thr, thr_err

        return best_thr, best_thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

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
        # Get the predictions
        y_pred = self._predict(X)

        # Calculate the misclassification error
        return misclassification_error(y, y_pred)
