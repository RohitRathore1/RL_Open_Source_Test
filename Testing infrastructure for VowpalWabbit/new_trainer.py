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