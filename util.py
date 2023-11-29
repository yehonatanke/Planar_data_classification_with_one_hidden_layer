import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import sklearn
import sklearn.datasets
import sklearn.linear_model
import copy


def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary.

    Parameters:
    - model (callable): A callable object representing the trained machine learning model.
    - X (numpy.ndarray): Input data with shape (n_features, n_samples) where n_features is 2.
    - y (numpy.ndarray): Target labels with shape (n_samples).

    Returns:
    None

    This function takes a trained machine learning model, input data, and target labels, and
    plots the decision boundary along with the training examples.

    Example:
    >>> # Assuming 'model' is a trained classifier and 'X', 'y' are training data
    >>> plot_decision_boundary(model, X, y)
    """

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def sigmoid(x):
    """
    Compute the sigmoid of x.

    Arguments:
    x -- A scalar or numpy array of any size.

    Returns:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def load_planar_dataset():
    """
    Load a simple planar dataset for binary classification.

    Returns:
    - X (numpy.ndarray): Feature matrix with dimensions (2, number of examples).
    - Y (numpy.ndarray): Label matrix with dimensions (1, number of examples).
    """
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    """
    Load additional datasets for testing.

    Returns:
    - noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure: Various datasets.
    """
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


def show_training_examples_info(X, Y):
    """
    Display information about the training examples.

    Parameters:
    - X (numpy.ndarray): Feature matrix with dimensions (number of examples, features).
    - Y (numpy.ndarray): Label matrix with dimensions (number of examples, labels).

    Returns:
    None

    Prints:
    - The shape of the feature matrix `X`.
    - The shape of the label matrix `Y`.
    - The total number of training examples `m`.
    """
    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[0]  # Number of training examples

    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('Number of training examples (m): %d' % m)


def train_logistic_regression(X, Y):
    """
    Train a logistic regression classifier using sklearn's LogisticRegressionCV.

    Parameters:
    - X (numpy.ndarray): Feature matrix with dimensions (number of examples, features).
    - Y (numpy.ndarray): Label matrix with dimensions (number of examples, labels).

    Returns:
    - clf (LogisticRegressionCV): Trained logistic regression classifier.
    """
    # Print shapes for debugging
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Ensure the number of samples is consistent
    if X.shape[0] != Y.shape[1]:
        raise ValueError("Inconsistent number of samples between X and Y")

    clf = LogisticRegressionCV()
    clf.fit(X.T, Y.ravel())  # Transpose X to match Y's shape

    return clf


def print_accuracy(predictions, Y):
    """
    Print the accuracy of a classifier based on its predictions.

    Parameters:
    - predictions (numpy.ndarray): Predicted labels.
    - Y (numpy.ndarray): True labels.

    Returns:
    None

    Prints:
    - Accuracy percentage.
    """
    accuracy = float(np.dot(Y, predictions) + np.dot(1 - Y, 1 - predictions)) / float(Y.size) * 100
    print('Accuracy: %.2f%% (percentage of correctly labelled datapoints)' % accuracy)


def layer_sizes(X, Y):
    """
    Defining the neural network structure.

    Arguments:
    - X (numpy.ndarray): Input dataset of shape (input size, number of examples).
    - Y (numpy.ndarray): Labels of shape (output size, number of examples).

    Returns:
    - n_x (int): Size of the input layer.
    - n_h (int): Size of the hidden layer.
    - n_y (int): Size of the output layer.
    """
    n_x = X.shape[0]  # Size of the input layer
    n_h = 4  # Size of the hidden layer (assuming 4 units)
    n_y = Y.shape[0]  # Size of the output layer

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize the parameters of a neural network.

    Arguments:
    - n_x (int): Size of the input layer.
    - n_h (int): Size of the hidden layer.
    - n_y (int): Size of the output layer.

    Returns:
    - parameters (dict): Python dictionary containing the initialized parameters.
                        - W1: Weight matrix of shape (n_h, n_x).
                        - b1: Bias vector of shape (n_h, 1).
                        - W2: Weight matrix of shape (n_y, n_h).
                        - b2: Bias vector of shape (n_y, 1).
    """
    # Initialize weights with small random values
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # Create a dictionary to store the parameters
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    Perform forward propagation in a neural network.

    Arguments:
    - X (numpy.ndarray): Input data of size (n_x, m).
    - parameters (dict): Python dictionary containing the parameters.

    Returns:
    - A2 (numpy.ndarray): The sigmoid output of the second activation.
    - cache (dict): A dictionary containing intermediate values "Z1", "A1", "Z2", and "A2".
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    # Store intermediate values in a cache dictionary
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):
    """
    Compute the Cost

    Arguments:
    - A2 (numpy.ndarray): The sigmoid output of the second activation, of shape (1, number of examples).
    - Y (numpy.ndarray): "True" labels vector of shape (1, number of examples).

    Formula:
    𝐽=−1𝑚∑𝑖=1𝑚(𝑦(𝑖)log(𝑎[2](𝑖))+(1−𝑦(𝑖))log(1−𝑎[2](𝑖)))

    Returns:
    - cost (float): Cross-entropy cost calculated using the sigmoid output and true labels.
    """
    m = Y.shape[1]  # number of examples

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
    cost = (-1 / m) * np.sum(logprobs)

    cost = float(np.squeeze(cost))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation for a neural network.

    Arguments:
    - parameters (dict): Python dictionary containing the parameters.
    - cache (dict): A dictionary containing intermediate values "Z1", "A1", "Z2", and "A2".
    - X (numpy.ndarray): Input data of shape (input size, number of examples).
    - Y (numpy.ndarray): "True" labels vector of shape (output size, number of examples).

    Returns:
    - grads (dict): Python dictionary containing gradients with respect to different parameters.
    """
    m = X.shape[1]

    # Retrieve W1, W2, A1, A2 from dictionaries "parameters" and "cache".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Update parameters using the gradient descent update rule.

    Arguments:
    - parameters (dict): Python dictionary containing the parameters.
    - grads (dict): Python dictionary containing the gradients.
    - learning_rate (float): The learning rate for gradient descent.

    Returns:
    - parameters (dict): Python dictionary containing the updated parameters.
    """
    # Retrieve a copy of each parameter from the dictionary "parameters".
    W1 = copy.deepcopy(parameters["W1"])
    b1 = copy.deepcopy(parameters["b1"])
    W2 = copy.deepcopy(parameters["W2"])
    b2 = copy.deepcopy(parameters["b2"])

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Build and train a neural network model.

    Arguments:
    - X (numpy.ndarray): Dataset of shape (input size, number of examples).
    - Y (numpy.ndarray): Labels of shape (output size, number of examples).
    - n_h (int): Size of the hidden layer.
    - num_iterations (int): Number of iterations in the gradient descent loop.
    - print_cost (bool): If True, print the cost every 1000 iterations.

    Returns:
    - parameters (dict): Parameters learned by the model, which can be used for prediction.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        # Cost function
        cost = compute_cost(A2, Y)
        # Backpropagation
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Use the learned parameters to predict a class for each example in X.

    Arguments:
    - parameters (dict): Python dictionary containing the parameters.
    - X (numpy.ndarray): Input data of size (input size, number of examples).

    Returns:
    - predictions (numpy.ndarray): Vector of predictions of our model (0: class 0 (red) / 1: class 1 (blue)).
    """

    # Compute probabilities using forward propagation and classify to 0/1 using 0.5 as the threshold
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions
