import unittest
from new_trainer import NewTrainer
import numpy as np

class TestNewTrainer(unittest.TestCase):
    def test_simple_train_and_predict(self):
        trainer = NewTrainer()
        X_train = [[1], [2], [3], [4]]
        y_train = [2, 4, 6, 8]
        trainer.train(X_train, y_train)
        X_test = [5]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 10, delta=0.1)

    def test_train_and_predict_multidimensional(self):
        trainer = NewTrainer()
        X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y_train = [3, 5, 7, 9]
        trainer.train(X_train, y_train)
        X_test = [5, 6]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 11, delta=0.1)

    def test_train_and_predict_noisy_data(self):
        trainer = NewTrainer()
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 3 * X_train.flatten() + 5 + np.random.normal(0, 1, 100)
        trainer.train(X_train.tolist(), y_train.tolist())
        X_test = [7]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 26, delta=1)

    def test_train_and_predict_large_input(self):
        trainer = NewTrainer()
        X_train = [[i] for i in range(1, 10001)]
        y_train = [i * 2 for i in range(1, 10001)]
        trainer.train(X_train, y_train)
        X_test = [10001]
        y_pred = trainer.predict(X_test)
        self.assertAlmostEqual(y_pred, 20002, delta=0.1)

    def test_train_and_predict_zero_length_input(self):
        trainer = NewTrainer()
        with self.assertRaises(ValueError):
            trainer.train([], [])

    def test_train_and_predict_mismatched_input_lengths(self):
        trainer = NewTrainer()
        X_train = [[1], [2], [3]]
        y_train = [2, 4]
        with self.assertRaises(ValueError):
            trainer.train(X_train, y_train)

    def test_train_and_predict_non_numerical_input(self):
        trainer = NewTrainer()
        X_train = [['a'], ['b'], ['c']]
        y_train = ['d', 'e', 'f']
        with self.assertRaises(ValueError):
            trainer.train(X_train, y_train)
            
    def test_predict_without_training(self):
        trainer = NewTrainer()
        X_test = [5]
        with self.assertRaises(RuntimeError):
            trainer.predict(X_test)

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