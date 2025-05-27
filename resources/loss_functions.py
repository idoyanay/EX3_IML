import numpy as np

def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays")

    # Calculate the number of misclassified samples
    misclassified = np.sum(y_true != y_pred)

    # Normalize by the number of samples if requested
    if normalize:
        return misclassified / len(y_true)
    else:
        return misclassified
    

