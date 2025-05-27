from typing import Tuple
import numpy as np
from base_estimator import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data. Has functions: fit, predict, loss

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    np.random.seed(42)  # For reproducibility
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    folds = np.array_split(indices, cv)

    train_scores = []
    validation_scores = []

    for i in range(cv):
        # Create train and validation indices using the folds
        val_indices = folds[i]
        train_indices = np.concatenate(folds[:i] + folds[i+1:])
        # Split the data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Fit the estimator
        estimator.fit(X_train, y_train)

        # Calculate scores
        train_score = estimator.loss(X_train, y_train)
        val_score = estimator.loss(X_val, y_val)

        train_scores.append(train_score)
        validation_scores.append(val_score)

    return np.mean(train_scores), np.mean(validation_scores)


    
