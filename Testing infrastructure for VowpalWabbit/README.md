## Screening Exercise

Let’s say we have just implemented a new training algorithm for regression 
with the following interface:

```python
class NewTrainer:
    ...
    def train(self, x: List[List[float]], y: List[float]):
        ...

    def predict(self, x: List[float]) -> float:
        ...
        return 0
```

Design and write test suite for it in Python using `unittest` or `pytest` 
frameworks.


## Solution

Here is a basic implementation of the `NewTrainer` class in a file called 
`new_trainer.py`. In this example, I've used a linear regression model from 
the `scikit-learn` library as the underlying model.

```python
from typing import List
from sklearn.linear_model import LinearRegression

class NewTrainer:
    def __init__(self):
        self.model = None

    def train(self, x: List[List[float]], y: List[float]):
        if len(x) != len(y):
            raise ValueError("Training input and output dimensions should be the same")

        self.model = LinearRegression()
        self.model.fit(x, y)

    def predict(self, x: List[float]) -> float:
        if self.model is None:
            raise RuntimeError("Model should be trained before making predictions")

        prediction = self.model.predict([x])
        return float(prediction[0])
```

This code defines a Python class called `NewTrainer` that uses the 
`LinearRegression` model from the `scikit-learn` library to perform 
linear regression. Linear regression is a machine learning method that 
predicts a continuous target variable `(y)` based on one or more input 
features `(x)`. 

Here's an example code of a test suite for the `NewTrainer` class using 
the `unittest` framework. The available tests in the test suite cover 
various scenarios for the `NewTrainer` class:

- `test_simple_train_and_predict`: Tests the basic functionality of 
   training the model with simple linear data and making a prediction.

- `test_train_and_predict_multidimensional`: Tests the training and 
   prediction with multidimensional input data.

- `test_train_and_predict_noisy_data`: Tests the model's ability to 
   train and predict on noisy data.

- `test_train_and_predict_large_input`: Tests the model's ability to 
   handle large input data during training and prediction.

- `test_train_and_predict_zero_length_input`: Checks that the model 
   raises a `ValueError` when attempting to train on empty input data.

- `test_train_and_predict_mismatched_input_lengths`: Ensures that the 
   model raises a `ValueError` when training with mismatched input and 
   output lengths.

- `test_train_and_predict_non_numerical_input`: Validates that the model 
   raises a `ValueError` when attempting to train with non-numerical input 
   data.

- `test_predict_without_training`: Checks that the model raises a 
   `RuntimeError` when attempting to make a prediction before training.

- `test_train_multiple_times_and_predict`: Tests the model's ability to 
   train on new data multiple times and make a prediction.

- `test_train_and_predict_with_negative_input`: Tests the model's ability 
   to train and predict with negative input values.

- `test_train_and_predict_non_linear_data`: Tests the model's ability to 
   train on non-linear data and make a prediction. Note that since the 
   underlying model is linear, the prediction will not be perfect.

- `test_train_and_predict_different_input_dimensions`: Ensures that the 
   model raises a ValueError when making a prediction with input data of 
   different dimensions than the training data.


```python
import unittest
from new_trainer import NewTrainer
import numpy as np

class TestNewTrainer(unittest.TestCase):
    # Tests the basic functionality of training the model with simple linear 
    # data and making a prediction.
    def test_simple_train_and_predict(self):
        trainer = NewTrainer()
        X_train = [[1], [2], [3], [4]]
        y_train = [2, 4, 6, 8]
        trainer.train(X_train, y_train)
        X_test = [5]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 10, delta=0.1)
    
    # Tests the training and prediction with multidimensional input data
    def test_train_and_predict_multidimensional(self):
        trainer = NewTrainer()
        X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y_train = [3, 5, 7, 9]
        trainer.train(X_train, y_train)
        X_test = [5, 6]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 11, delta=0.1)
    
    # Tests the model's ability to train and predict on noisy data
    def test_train_and_predict_noisy_data(self):
        trainer = NewTrainer()
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 3 * X_train.flatten() + 5 + np.random.normal(0, 1, 100)
        trainer.train(X_train.tolist(), y_train.tolist())
        X_test = [7]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 26, delta=1)
    
    # Tests the model's ability to handle large input data during training 
    # and prediction
    def test_train_and_predict_large_input(self):
        trainer = NewTrainer()
        X_train = [[i] for i in range(1, 10001)]
        y_train = [i * 2 for i in range(1, 10001)]
        trainer.train(X_train, y_train)
        X_test = [10001]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 20002, delta=0.1)
    
    # Checks that the model raises a ValueError when attempting to train on 
    # empty input data
    def test_train_and_predict_zero_length_input(self):
        trainer = NewTrainer()
        with self.assertRaises(ValueError):
            trainer.train([], [])
    
    # Ensures that the model raises a ValueError when training with mismatched 
    # input and output lengths
    def test_train_and_predict_mismatched_input_lengths(self):
        trainer = NewTrainer()
        X_train = [[1], [2], [3]]
        y_train = [2, 4]
        with self.assertRaises(ValueError):
            trainer.train(X_train, y_train)
    
    # Validates that the model raises a ValueError when attempting to train with 
    # non-numerical
    def test_train_and_predict_non_numerical_input(self):
        trainer = NewTrainer()
        X_train = [['a'], ['b'], ['c']]
        y_train = ['d', 'e', 'f']
        with self.assertRaises(ValueError):
            trainer.train(X_train, y_train)

    # Checks that the model raises a RuntimeError when attempting to make a 
    # prediction before training        
    def test_predict_without_training(self):
        trainer = NewTrainer()
        X_test = [5]
        with self.assertRaises(RuntimeError):
            trainer.predict(X_test)
    
    # Tests the model's ability to train on new data multiple times
    def test_train_multiple_times_and_predict(self):
        trainer = NewTrainer()
        X_train = [[1], [2], [3], [4]]
        y_train = [2, 4, 6, 8]
        trainer.train(X_train, y_train)

        X_train2 = [[4], [5], [6], [7]]
        y_train2 = [8, 10, 12, 14]
        trainer.train(X_train2, y_train2)

        X_test = [8]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 16, delta=0.1)

    # Tests the model's ability to train and predict with negative input values
    def test_train_and_predict_with_negative_input(self):
        trainer = NewTrainer()
        X_train = [[-2], [-1], [0], [1], [2]]
        y_train = [-4, -2, 0, 2, 4]
        trainer.train(X_train, y_train)
        X_test = [-3]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, -6, delta=0.1)
   
    def test_train_and_predict_non_linear_data(self):
        trainer = NewTrainer()
        X_train = [[1], [2], [3], [4]]
        y_train = [1, 4, 9, 16]  # Quadratic relationship
        trainer.train(X_train, y_train)
        X_test = [5]
        y_pred = trainer.predict(X_test)
        # Note that the linear model will not fit the data perfectly,
        # so we need to increase the delta for the assertion
        self.assertAlmostEqual(y_pred, 25, delta=5)

    def test_train_and_predict_different_input_dimensions(self):
        trainer = NewTrainer()
        X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y_train = [3, 5, 7, 9]
        trainer.train(X_train, y_train)
        X_test = [5, 6, 7]  # Different input dimensions
        with self.assertRaises(ValueError):
            trainer.predict(X_test)

if __name__ == '__main__':
    unittest.main()
```

These tests provide a comprehensive suite to ensure the correct behavior of the 
`NewTrainer` class across various situations. The tests are run by the `unittest`.

## Instructions

```
python -m unittest test_new_trainer.py
```