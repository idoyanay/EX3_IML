import numpy as np
import plotly.graph_objects as go
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression

def q1(n_samples: int = 50, n_evaluations: int = 500):
    """
    Load the diabetes dataset and split it to a training and testing portion, using only 50 random
    samples for training (So, we can see the strength of k-fold cross-validation on such a small
    training set). The remaining samples should be used as a fixed test set for Question 3.

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate for training
    n_evaluations: int, default=500
        Number of regularization parameter values to evaluate for each of the algorithms
    
    Returns
    -------
    X_train: ndarray of shape (n_samples, n_features)
    y_train: ndarray of shape (n_samples, )
        Training data and labels
    X_test: ndarray of shape (n_samples, n_features)
    y_test: ndarray of shape (n_samples, )
        Test data and labels
    """

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Randomly select n_samples from the dataset
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_train = X[indices]
    y_train = y[indices]

    X_test = np.delete(X, indices, axis=0)
    y_test = np.delete(y, indices, axis=0)

    return X_train, y_train, X_test, y_test


def q2(X_train: np.ndarray, y_train: np.ndarray, n_evaluations: int = 500):
    """ For both Ridge and Lasso regularizations (both with intercept), run 5-Fold Cross-Validation
    using different values of the regularization parameter λ. Explore possible ranges of values of
    λ (say, take n_evaluations=500 equally spaced values in a range of your choice). Which
    range of values is meaningful for the given data for each of the algorithms?
    Parameters
    ----------
    X_train: ndarray of shape (n_samples, n_features)
        Training data
    y_train: ndarray of shape (n_samples, )
        Training labels
    n_evaluations: int, default=500
        Number of regularization parameter values to evaluate for each of the algorithms

    Returns
    -------
    best_ridge_lambda: float
        Best regularization parameter for Ridge Regression
    best_lasso_lambda: float
        Best regularization parameter for Lasso Regression
    """
    # Define the range of regularization parameters
    ridge_lambdas = np.linspace(0.1 ** 3, 10, n_evaluations)
    lasso_lambdas = np.linspace(0.1 ** 3, 5, n_evaluations)

    ridge_results = []
    lasso_results = []

    # Perform cross-validation for Ridge Regression
    for lam in ridge_lambdas:
        ridge_model = RidgeRegression(lam=lam, include_intercept=True)
        train_score, val_score = cross_validate(ridge_model, X_train, y_train, cv=5)
        ridge_results.append((train_score, val_score))

    # Perform cross-validation for Lasso Regression
    for lam in lasso_lambdas:
        lasso_model = Lasso(alpha=lam, include_intercept=True)
        train_score, val_score = cross_validate(lasso_model, X_train, y_train, cv=5)
        lasso_results.append((train_score, val_score))

    # Find the best regularization parameters based on validation scores
    best_ridge_lambda = ridge_lambdas[np.argmin([val for _, val in ridge_results])]
    best_lasso_lambda = lasso_lambdas[np.argmin([val for _, val in lasso_results])]
    
    ridge_train_scores, ridge_val_scores = zip(*ridge_results)
    lasso_train_scores, lasso_val_scores = zip(*lasso_results)
    
    # plot the results for the Ridge Regression
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ridge_lambdas, y=ridge_train_scores, mode='lines', name='Ridge Train Score'))
    fig.add_trace(go.Scatter(x=ridge_lambdas, y=ridge_val_scores, mode='lines', name='Ridge Validation Score'))
    fig.update_layout(title='Ridge Regression: Train and Validation Scores vs Regularization Parameter',
                        xaxis_title='Regularization Parameter (λ)',
                        yaxis_title='MSE Loss',
                        legend=dict(x=0, y=1))
    fig.show()
    # plot the results for the Lasso Regression
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lasso_lambdas, y=lasso_train_scores, mode='lines', name='Lasso Train Score'))
    fig.add_trace(go.Scatter(x=lasso_lambdas, y=lasso_val_scores, mode='lines', name='Lasso Validation Score'))
    fig.update_layout(title='Lasso Regression: Train and Validation Scores vs Regularization Parameter',
                        xaxis_title='Regularization Parameter (λ)',
                        yaxis_title='MSE Loss',
                        legend=dict(x=0, y=1))
    fig.show()

    return best_ridge_lambda, best_lasso_lambda


def q3(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
        best_ridge_lambda: float, best_lasso_lambda: float):
    """
    Compare the best Ridge model, best Lasso model and Least Squares model on the test set.
    Parameters
    ----------
    X_train: ndarray of shape (n_samples, n_features)
        Training data
    y_train: ndarray of shape (n_samples, )
        Training labels
    X_test: ndarray of shape (n_samples, n_features)
    y_test: ndarray of shape (n_samples, )
        Test data and labels
    best_ridge_lambda: float
        Best regularization parameter for Ridge Regression
    best_lasso_lambda: float 
    """
    print(f'---- Best Ridge Lambda: {best_ridge_lambda:.4f} ----')
    print(f'---- Best Lasso Lambda: {best_lasso_lambda:.4f} ----')
    # Fit the best Ridge model
    ridge_model = RidgeRegression(lam=best_ridge_lambda, include_intercept=True)
    ridge_model.fit(X_train, y_train)
    ridge_test_loss = ridge_model.loss(X_test, y_test)
    print(f'---- Ridge Regression Test Loss: {ridge_test_loss:.4f} ----')
    # Fit the best Lasso model
    lasso_model = Lasso(alpha=best_lasso_lambda, include_intercept=True)
    lasso_model.fit(X_train, y_train)
    lasso_test_loss = lasso_model.loss(X_test, y_test)
    print(f'---- Lasso Regression Test Loss: {lasso_test_loss:.4f} ----')
    # Fit the Least Squares model
    least_squares_model = LinearRegression()
    least_squares_model.fit(X_train, y_train)
    least_squares_test_loss = least_squares_model.loss(X_test, y_test)
    print(f'---- Least Squares Regression Test Loss: {least_squares_test_loss:.4f} ----')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X_train, y_train, X_test, y_test = q1(n_samples, n_evaluations)

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    best_ridge_lambda, best_lasso_lamda = q2(X_train, y_train, n_evaluations)

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    q3(X_train, y_train, X_test, y_test, best_ridge_lambda, best_lasso_lamda)


if __name__ == '__main__':
    np.random.seed(42)
    select_regularization_parameter(n_samples=50, n_evaluations=500)
