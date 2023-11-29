from util import *
# Planar_data_classification_with_one_hidden_layer
# The general methodology to build a Neural Network is to:
#   1. Define the neural network structure (# of input units, # of hidden units, etc).
#   2. Initialize the model's parameters
#   3. Loop:
#       - Implement forward propagation
#       - Compute loss
#       - Implement backward propagation to get the gradients
#       - Update parameters (gradient descent)

# Load the dataset
X, Y = load_planar_dataset()

# Visualize the dataset using matplotlib. The data looks like a "flower" with some red (label y=0) and some
# blue (y=1) points. The program goal is to build a model to fit this data. In other words, we want the classifier
# to define regions as either red or blue.
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors='k', s=40);
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of Data Points')
plt.show()

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()

# Build a model with n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# Print accuracy
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
plt.figure(figsize=(16, 32))

# you can try with different hidden layer sizes
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.show()
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
