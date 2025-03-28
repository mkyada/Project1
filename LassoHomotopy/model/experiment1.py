import numpy as np
import matplotlib.pyplot as plt
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel
from generate_regression_data import linear_data_generator

def run_experiment_generated():
    # Generate synthetic regression data using the provided generator.
    # For example, let true coefficients be [1.5, -2.0, 0.5] and intercept 2.0.
    m = np.array([1.5, -2.0, 0.5])
    b = 2.0
    X, y = linear_data_generator(m, b, [-5, 5], 200, 1.0, 123)
    
    # Fit the LassoHomotopy model.
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    
    # Plot the regularization path.
    results.plot_path()
    
    # Plot true versus predicted values.
    y_pred = results.predict(X)
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.7)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Predicted vs True Values")
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.show()

if __name__ == "__main__":
    run_experiment_generated()
