# Team Members

| Name             | Student ID | Email                  |
| ---------------- | ---------- | ---------------------- |
| Minal Kyada | A20544029  | mkyada@hawk.iit.edu  |
| Bhavikk Shah | A20543706 | bshah49@hawk.iit.edu |
| Manan Shah | A20544907 | mshah130@hawk.iit.edu   |

# How to Run

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/mkyada/Project1.git
   cd Project1/
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate Virtual Environment

   **Linux/Mac:**

   ```bash
   source venv/bin/activate
   ```

4. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the data generation script:
    
     ```bash
   cd LassoHomotopy/models/
   python experiment1.py
   ```

6. Change Directory to test folder

   ```bash
   cd LassoHomotopy/tests/
   ```

7. Execute pytest for testing the code

   ```bash
   pytest -s
   ```

# Frequently Asked Questions.

1. **What does the model you have implemented do and when should it be used ?**

    - The model we built is a LASSO regression solver that uses something called the Homotopy Method. In simple terms, it's a tool that helps identify which features in a dataset are truly important by shrinking some of the less useful ones down to zero. 
   
    - This is especially helpful when you're working with datasets that have a lot of features (columns) and you want a model that’s both simple and easy to interpret. 
   
    - You'd typically use this when you care about understanding which inputs really matter rather than just making predictions.

2. **How did you test your model to determine if it is working reasonably correctly ?**

    - We tested the model using a set of unit tests written in a file called test_LassoHomotopy.py. 
    
    - These tests check whether the model behaves as expected in different scenarios—for example, when features in the data are highly correlated (as in collinear_data.csv), the model should still be able to produce a sparse and meaningful solution.
    
    - To make sure the results make sense, we also checked key metrics like Mean Squared Error (MSE) to ensure it's never negative, and R² scores to confirm they're within a reasonable range. 
    
    - We also used visualizations, like comparing predicted values to actual ones in a scatter plot and plotting the learned coefficients, to further confirm that the model is working correctly.

3. **What parameters have you exposed to users of your implementation in order to tune performance ?**

    - Users can tweak a few settings to control how the model behaves:
        - alpha: This sets how strongly we penalize large coefficients. A higher alpha will make the model more selective and drop more features.
        
        - tol (tolerance): This determines how precise the solution should be. Lowering it gives a more accurate result but can take longer to compute.
        
        - max_iter: This sets a limit on how many steps the algorithm takes, which helps prevent it from running forever if it's taking too long to converge.

4. **Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental ?**

Yes, there are a few situations where the current model can face challenges:

    - Large datasets: Because the Homotopy Method can be computationally heavy, performance may slow down when there are a lot of features.

    - Unscaled data: The model assumes that all features are on the same scale. If the data isn’t standardized, the results may be off.

    - Highly correlated features: Although LASSO is designed to handle correlated features, it can struggle when multiple features are equally important—it might pick one and ignore others arbitrarily.

    If We had more time, We could improve these issues by using dimensionality reduction techniques like PCA or trying out other optimization methods that scale better for large datasets.

# Lasso Homotopy Regression Project

This project provides an implementation of the Lasso regression model using the Homotopy method—an efficient technique for solving L1-regularized least squares problems. It includes tools for generating synthetic data, the core logic of the Homotopy-based Lasso algorithm, and a robust suite of test cases to ensure the model performs accurately across different scenarios.

## Table of Contents

1. [Architecture](#architecture)
2. [Implementation Details](#implementation-details)
3. [Usage Examples](#usage-examples)
4. [Test Cases](#test-cases)
5. [Dependencies](#dependencies)
6. [Sample Code](#sample-code)

## Architecture

The project is organized into the following structure:

```
.
├── generate_regression_data.py    # Data generation utility
├── requirements.txt              # Project dependencies
└── LassoHomotopy/
    ├── model/
    │   └── LassoHomotopy.py     # Core implementation
    │   └── experiment1.py       # visual representation of comaprison graphs
    └── tests/
        ├── test_LassoHomotopy.py # Test suite 1
        ├── test_cases.py   # Test suite 2
        ├── small_test.csv        # Test dataset
        └── collinear_data.csv    # Test dataset
```

### Key Components

1. **Synthetic Data Generator (`generate_regression_data.py`)**

   - Creates customizable synthetic datasets for regression tasks.
   - Allows control over noise levels, number of features, and sample size.
   - Saves the generated datasets in CSV format for easy reuse

2. **Lasso Regression with Homotopy (`LassoHomotopy.py`)**

   - Implements the Lasso regression algorithm using the Homotopy approach.
   - Includes feature standardization for consistent performance.
   - Offers fit and predict methods for training and inference.

3. **Testing and Evaluation (`test_LassoHomotopy.py`)**

   - Contains a wide range of test cases to ensure model correctness
   - Evaluates model accuracy using metrics like MSE and R²
   - Includes visual tools to assess model predictions and coefficient sparsity across various datasets

## Implementation Details

### Lasso Homotopy Algorithm

This project implements Lasso regression using the Homotopy method, a stepwise algorithm designed to efficiently solve L1-regularized least squares problems. The method progressively builds the solution path, making it especially suitable for problems where feature selection and model interpretability are important.

The model solves the belo optimization problem
```
min ||y - Xβ||²₂ + α||β||₁
```

# Key Highlights
**Automatic Feature Selection:** Encourages sparsity by shrinking less useful coefficients to zero.

**Regularization Path Tracking:** Traces how coefficients evolve as the regularization strength (lambda) decreases.

**Numerical Stability:** Handles edge cases such as zero-variance features or highly collinear data.

# Core Components:

1. **LassoHomotopyModel**

- Implements the Homotopy-based Lasso regression algorithm.

- Standardizes input data and centers the target variable.

- Tracks the evolution of coefficients and λ across iterations.

- Uses the equiangular vector approach to identify step directions.

2. **LassoHomotopyResults**

   - Handles model outputs:
        coef: Final coefficient values.
        intercept: Learned intercept term.
        predict(X): Makes predictions on new data.
        plot_path(): Visualizes how coefficients change as regularization decreases.


## Usage Examples

### 1. Generating Synthetic Data

```python
python generate_regression_data.py -N 1000 -m 1.0 0.5 -0.3 -b 2.0 -scale 0.1 -rnge -5 5 -seed 42 -output_file data.csv
```

### 2. Training the Lasso Model

```python
from LassoHomotopy import LassoHomotopyModel

# Sample input
import pandas as pd
df = pd.read_csv('data.csv')
X = df.filter(regex='^x_').values
y = df['y'].values

# Train model
model = LassoHomotopyModel()
results = model.fit(X, y)

# Predict
predictions = results.predict(X)

# Visualize coefficient path
results.plot_path()

```

# Parameters
**max_steps:** Maximum number of steps in the Homotopy path (default = 1000).

**tol:** Tolerance for convergence and numerical stability (default = 1e-6).

# Known Limitations
Scalability: The algorithm may be slower on datasets with a very large number of features.

Standardization Required: Input features must be scaled for meaningful results.

Correlated Features: May arbitrarily pick one from a group of equally relevant features.

# Ideal use cases
High-dimensional datasets where feature selection is important.

Situations requiring interpretable models.

Academic or research settings focused on understanding model behavior.

## Test Cases

The test suite covers multiple scenarios and generates detailed visualizations for analysis. Running `pytest -s` produces the following results:

### Testing & Evaluation
This project includes an extensive test suite designed to ensure the correctness, robustness, and performance of the Lasso Homotopy model across a variety of datasets and edge cases.

## Core Test Results

1. **Small Dataset Evaluation (`small_test.csv`)**
- Mean Predicted Value: ~3.3350 (Verified against known output)

- Confirms that the model can learn accurately from small, simple datasets.

2. **Collinear Data Evaluation (`collinear_data.csv`)**
- Model successfully eliminates redundant features, confirming its capability to handle multicollinearity.

- At least one coefficient is effectively shrunk to ~0, demonstrating Lasso’s sparsity behavior.

3. **Synthetic Dataset Experiment**
- Using `experiment1.py`, a synthetic regression dataset was generated with known coefficients.

- The model closely matches true values and plots a well-behaved regularization path and prediction vs. actual values, visually confirming correctness.

### Generated Visualizations

The graph shows the comparison between Scikit Learn's Lasso Model and Custom Lasso Model wrt the true value of y: 


### Test Coverage

The test suite validates the model across different scenarios:

✔️ Functionality Tests (test_cases.py)
Intercept accuracy (constant values, scaling validation)
Sparsity under collinearity (collinear feature elimination)
Shape and output checks (prediction dimensions, path length)
Edge cases: all-zero inputs, more features than samples, low noise (overfitting detection)

✔️ Regression Path Tests
Ensures that the λ (lambda) values in the regularization path decrease monotonically, as expected from the Homotopy method.

✔️ Consistency Tests
Running the model multiple times on the same dataset produces the same coefficients and intercept, ensuring deterministic behavior.

✔️ Visualization Validation (experiment1.py)
Validates that plots (like regularization paths and prediction scatter plots) generate without errors and are informative for model diagnostics.

### Example Parametrized Test Structure

This enables flexible testing across different datasets, automating checks for accuracy, reliability, and numerical correctness: 

```python
@pytest.mark.parametrize("csv_path", ["small_test.csv", "collinear_data.csv"])
def test_lasso_model(csv_path):
    # Loads dataset, fits model, validates prediction and coefficients
```

## Dependencies

Required packages (specified versions in requirements.txt):

```
numpy
scipy
pandas
matplotlib
pytest
scikit-learn
```

## Sample Code

### Feature Importance Analysis

```python
class LassoHomotopyResults():
    def __init__(self, coef, intercept, path_lambdas, path_betas):
        self.coef = coef
        self.intercept = intercept
        self.path_lambdas = path_lambdas
        self.path_betas = path_betas

    def predict(self, X):
        X = np.array(X, dtype=float)
        return self.intercept + X.dot(self.coef)

    def plot_path(self):
        # Plot the evolution of each coefficient versus lambda.
        path_betas = np.array(self.path_betas)
        lambdas = self.path_lambdas
        plt.figure()
        for i in range(path_betas.shape[1]):
            plt.plot(lambdas, path_betas[:, i], label=f'coef_{i}')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficient value')
        plt.title('Regularization Path')
        plt.legend()
        plt.gca().invert_xaxis()  # since lambda decreases over iterations
        plt.show()
```