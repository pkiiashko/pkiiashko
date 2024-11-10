from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Fitting descent weights for x and y dataset
        :param x: features array
        :param y: targets array
        :return: self
        """
        # TODO: fit weights to x and y
        # Инициализация веса (обычно инициализируется нулями или случайными значениями)
        self.descent.init_weights(x.shape[1])

        for iteration in range(self.max_iter):
        # Получаем текущие предсказания модели
            predictions = self.predict(x)
        
        # Вычисляем текущую ошибку (потери) и сохраняем её в историю
            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)

        # Проверяем условие остановки по разнице в потере (критерий сходимости)
            if iteration > 0 and abs(self.loss_history[-2] - loss) < self.tolerance:
                print(f'Converged at iteration {iteration}, loss: {loss}')
                break

        # Обновляем веса с помощью градиентного спуска
            self.descent.step(x, y, predictions)

            return self
        raise NotImplementedError('LinearRegression fit function not implemented')

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
