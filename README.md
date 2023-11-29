# Planar Data Classification with One Hidden Layer

This Python program implements a simple neural network to classify planar data with one hidden layer. The overall methodology involves defining the neural network structure, initializing parameters, and performing iterative steps of forward propagation, loss computation, backward propagation for gradients, and parameter updates using gradient descent.

## Features
- **Dataset Handling:** Loads a planar dataset, separating features and labels using the `load_planar_dataset` function.

- **Visualization and Baseline:** Utilizes matplotlib to create a scatter plot of the dataset, trains a logistic regression classifier, and plots the decision boundary for baseline comparison.

- **Neural Network Modeling:** Defines and trains a neural network model with a hidden layer, visualizing its decision boundary and printing accuracy, offering flexibility for experimentation with different hidden layer sizes.

## Steps:

1. **Load Dataset:**
   - Utilizes a function `load_planar_dataset()` from the `util` module to load planar dataset (`X` features, `Y` labels).

2. **Visualize Dataset:**
   - Uses matplotlib to create a scatter plot of the dataset, where red points correspond to `y=0` and blue points correspond to `y=1`.

```python
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors='k', s=40);
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of Data Points')
plt.show()
```
3. **Train Logistic Regression Classifier:**
   - Uses scikit-learn's LogisticRegressionCV to train a logistic regression classifier and plots the decision boundary.

```python
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
```
4. **Build Neural Network Model:**
   - Builds a neural network model with a hidden layer of size n_h=4 using the nn_model function.

```python
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
```
5. **Visualize Decision Boundary:**
   - Plots the decision boundary for the neural network model.

```python
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
```

6. **Print Accuracy:**
   - Prints the accuracy of the model on the provided dataset.

```python
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
```
7. **Explore Different Hidden Layer Sizes:**
   - Iterates through different hidden layer sizes, builds models, plots decision boundaries, and prints accuracies.

```python
# Example with different hidden layer sizes
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
# ...
```

## Acknowledgments

This project is based on the materials provided in the 'Deep Learning Specialization' course on Coursera. The foundational concepts, logic, and inspiration for this project are derived from the course materials created by the DeepLearning.AI team.

#### Attribution

Course: [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning) 

Instructor: Andrew Ng 

Platform: [Coursera](https://www.coursera.org)

Host: [DeepLearning.AI](https://www.deeplearning.ai)

Please note that while the code structure has been extensively modified, the core ideas and problem-solving approach are based on the educational content provided in the course.

## License

This program is released under the [MIT License](https://github.com/yehonatanke/Planar_data_classification_with_one_hidden_layer/blob/main/LICENSE).

## Author

yehonataKe
