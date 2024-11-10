from __future__ import annotations
from typing import List
import numpy as np
from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    def __init__(self, descent_config: dict, tolerance: float = 1e-6, max_iter: int = 1000):
        self.descent = get_descent(descent_config)  # Получаем объект спуска
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_history = []  # История значений функции потерь

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Обучение модели с использованием градиентного спуска.
        :param x: Признаки
        :param y: Целевые значения
        :return: self
        """
        for iteration in range(self.max_iter):
            # Обновляем веса с помощью градиентного спуска (теперь без predictions)
            weight_diff = self.descent.step(x, y)
            loss = self.descent.calc_loss(x, y)
            self.loss_history.append(loss)

            # Проверка на сходимость
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break

        # После завершения обучения нужно добавить финальную потерю, если она еще не была добавлена
        if len(self.loss_history) < self.max_iter:
            self.loss_history.append(loss)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        return self.descent.calc_loss(x, y)

