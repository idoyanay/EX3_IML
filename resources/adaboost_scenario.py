import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def q1(train_X, train_y, test_X, test_y, n_learners, noise):
    """
    Question 1: Train- and test errors of AdaBoost in noiseless case
    """
    # Initialize AdaBoost with Decision Stump as base estimator
    ada = AdaBoost(DecisionStump, iterations=n_learners)
    
    # Fit the model on training data
    ada._fit(train_X, train_y)
    
    # calculate the training and test errors as function of the number of learners
    train_errors = []
    test_errors = []
    for t in range(1, n_learners + 1):
        train_errors.append(ada.partial_loss(train_X, train_y, t))
        test_errors.append(ada.partial_loss(test_X, test_y, t))
    
    # Plot the training and test errors
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=train_errors, mode='lines+markers', name='Train Error'))
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=test_errors, mode='lines+markers', name='Test Error'))
    fig.update_layout(title=f"Train and Test Errors of AdaBoost, Noise Ratio: {noise}",
                      xaxis_title='Number of Learners',
                      yaxis_title='Error',
                      legend=dict(x=0, y=1, traceorder='normal', orientation='h'))
    fig.show()

    return ada


def q2(ada, train_X, train_y, test_X, test_y):
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Ensemble size: {t}" for t in T])

    for i, t in enumerate(T):
        predict_lambda = lambda X : ada.partial_predict(X, t)
        row, col = divmod(i, 2)
        fig.add_trace(
            decision_surface(predict_lambda, lims[0], lims[1], showscale=False),
            row=row + 1, col=col + 1
        )
        # Add test set points with vivid symbol and color per label
        pos_mask = test_y == 1
        neg_mask = test_y == -1

        fig.add_trace(
            go.Scatter(
                x=test_X[pos_mask, 0],
                y=test_X[pos_mask, 1],
                mode='markers',
                name='+1',
                marker=dict(color='red', symbol='circle', size=6, line=dict(width=1, color='black')),
                showlegend=(i == 0)
            ),
            row=row + 1, col=col + 1
        )
        fig.add_trace(
            go.Scatter(
                x=test_X[neg_mask, 0],
                y=test_X[neg_mask, 1],
                mode='markers',
                name='-1',
                marker=dict(color='blue', symbol='x', size=8, line=dict(width=1, color='black')),
                showlegend=(i == 0)
            ),
            row=row + 1, col=col + 1
        )
        fig.update_xaxes(title_text='Feature 1', row=row + 1, col=col + 1)
        fig.update_yaxes(title_text='Feature 2', row=row + 1, col=col + 1)
    fig.update_layout(title=f"Decision Surfaces of AdaBoost Ensemble.",
                      height=800, width=800)
    fig.show()
    

def q3(ada, train_X, train_y, test_X, test_y):
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # find the best T
    best_test_error, t = np.inf, None
    for t in range(1, ada.iterations_ + 1):
        current_test_error = ada.partial_loss(test_X, test_y, t)
        if current_test_error < best_test_error:
            best_test_error = current_test_error
            best_t = t
    # convert the best_test_error to accuracy
    best_test_error = 1 - best_test_error

    predict_lambda = lambda X: ada.partial_predict(X, best_t)
    fig = go.Figure()
    fig.add_trace(decision_surface(predict_lambda, lims[0], lims[1], showscale=False))
    # Add test set points with vivid symbol and color per label
    pos_mask = test_y == 1
    neg_mask = test_y == -1

    fig.add_trace(
        go.Scatter(
            x=test_X[pos_mask, 0],
            y=test_X[pos_mask, 1],
            mode='markers',
            name='+1',
            marker=dict(color='red', symbol='circle', size=6, line=dict(width=1, color='black')),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=test_X[neg_mask, 0],
            y=test_X[neg_mask, 1],
            mode='markers',
            name='-1',
            marker=dict(color='blue', symbol='x', size=8, line=dict(width=1, color='black')),
        ),
    )
    fig.update_xaxes(title_text='Feature 1')
    fig.update_yaxes(title_text='Feature 2')
    fig.update_layout(title=f"Ensemble size: {best_t}, Test Error: {best_test_error:.4f}",
                      height=800, width=800)
    fig.show()

    return best_t


def q4(ada, train_X, train_y, test_X, test_y, noise):
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # Get the weights of the last iteration
    D = ada.D_[-1]
    D = D / np.max(D) * 5  # Normalize and scale the weights

    fig = go.Figure()
    fig.add_trace(decision_surface(ada._predict, lims[0], lims[1], showscale=False))

    # Add training set points with size proportional to their weight and color per label
    pos_mask = train_y == 1
    neg_mask = train_y == -1

    fig.add_trace(
        go.Scatter(
            x=train_X[pos_mask, 0],
            y=train_X[pos_mask, 1],
            mode='markers',
            name='+1',
            marker=dict(color='red', symbol='circle', size=D[pos_mask] , line=dict(width=1, color='black')),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=train_X[neg_mask, 0],
            y=train_X[neg_mask, 1],
            mode='markers',
            name='-1',
            marker=dict(color='blue', symbol='x', size=D[neg_mask] , line=dict(width=1, color='black')),
        ),
    )
    
    fig.update_xaxes(title_text='Feature 1')
    fig.update_yaxes(title_text='Feature 2')
    fig.update_layout(title=f"Decision Surface with Weighted Samples. Noise Ratio: {noise}",
                      height=800, width=800, legend=dict(font=dict(size=16),x=0.85, y=0.95))
    fig.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = q1(train_X, train_y, test_X, test_y, n_learners, noise)
    
    if noise == 0.0: # only run the following quesitons for the noiseless case
        # Question 2: Plotting decision surfaces
        q2(ada, train_X, train_y, test_X, test_y)

        # Question 3: Decision surface of best performing ensemble
        best_num_iter = q3(ada, train_X, train_y, test_X, test_y)

    # Question 4: Decision surface with weighted samples
    q4(ada, train_X, train_y, test_X, test_y, noise)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.0, n_learners=250, train_size=5000, test_size=500)
    fit_and_evaluate_adaboost(noise=0.4, n_learners=250, train_size=5000, test_size=500)