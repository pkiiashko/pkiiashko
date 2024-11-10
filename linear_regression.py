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
        self.previous_w = None  # Для отслеживания изменения весов

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Обучение модели с использованием градиентного спуска.
        :param x: Признаки
        :param y: Целевые значения
        :return: self
        """
        # Добавляем начальное значение потерь до первого шага
        loss = self.descent.calc_loss(x, y)
        self.loss_history.append(loss)
        
        for iteration in range(self.max_iter):
            # Обновляем веса с помощью градиентного спуска
            weight_diff = self.descent.step(x, y)
            loss = self.descent.calc_loss(x, y)
            self.loss_history.append(loss)
            
            # Проверка критериев останова
            if self.check_convergence():
                print(f"Converged at iteration {iteration}")
                break

        return self

    def check_convergence(self) -> bool:
        """
        Проверка критериев останова:
        1. Норма разности весов меньше tolerance.
        2. Наличие NaN в весах.
        3. Достигнут максимум итераций.
        """
        if self.previous_w is None:
            self.previous_w = self.descent.w.copy()
            return False
        
        # 1. Квадрат евклидовой нормы разности весов на двух соседних итерациях
        weight_diff_norm = np.linalg.norm(self.descent.w - self.previous_w)
        if weight_diff_norm < self.tolerance:
            return True
        
        # 2. Проверка на NaN в весах
        if np.any(np.isnan(self.descent.w)):
            return True
        
        # Обновляем предыдущее значение весов
        self.previous_w = self.descent.w.copy()
        return False

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
