from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        # Убедимся, что dimension — это целое число
        if isinstance(dimension, float):  # Проверяем, если это float
            dimension = int(dimension)  # Преобразуем в int

        self.dimension = dimension  # Инициализируем размерность как целое число

        # Инициализируем веса
        self.init_weights(self.dimension)

        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def init_weights(self, dimension: int) -> None:
        """Инициализация весов нулями."""
        if dimension > 0:
            self.w = np.zeros(dimension)  # Инициализируем веса нулями

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Шаг градиентного спуска.
        :param x: Признаки (features)
        :param y: Целевые значения (targets)
        :return: Разница весов (w_{k + 1} - w_k)
        """
        gradient = self.calc_gradient(x, y)
        return self.update_weights(gradient)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        # Линейная регрессия: y_pred = Xw
        y_pred = np.dot(x, self.w)  # Убедитесь, что self.w имеет правильную размерность
        return y_pred

class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class with learning rate decay
    """

    def __init__(self, learning_rate: float = 0.01, dimension: int = None, lambda_: float = 0.01,
                 s0: float = 1, p: float = 0.5, loss_function: callable = None, **kwargs):
        """
        Инициализация класса
        :param learning_rate: начальная скорость обучения (по умолчанию 0.01)
        :param dimension: размерность данных (по умолчанию None)
        :param lambda_: коэффициент регуляризации (по умолчанию 0.01)
        :param s0: параметр для вычисления скорости обучения (по умолчанию 1)
        :param p: параметр для вычисления скорости обучения (по умолчанию 0.5)
        :param loss_function: функция потерь (по умолчанию None)
        :param kwargs: дополнительные параметры
        """
        # Инициализация базового класса
        super().__init__(dimension=dimension, **kwargs)
        
        self.learning_rate = learning_rate  # Начальная скорость обучения
        self.lambda_ = lambda_  # Коэффициент регуляризации
        self.s0 = s0  # Параметр s0 для вычисления eta
        self.p = p  # Параметр p для вычисления eta
        self.loss_function = loss_function  # Функция потерь
        self.k = 0  # Счетчик итераций (для вычисления eta)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновить веса с учетом градиента
        :param gradient: градиент функции потерь
        :return: разница весов (w_{k + 1} - w_k): np.ndarray
        """
        # Вычисляем длину шага по формуле
        eta_k = self.learning_rate / (self.s0 + self.k) ** self.p  # Используем learning_rate

        # Рассчитываем разницу весов
        weight_diff = -eta_k * gradient  # Понижение по антиградиенту

        # Обновляем веса модели
        self.w -= weight_diff  # Обновление весов (w_{k + 1} = w_k - eta_k * gradient)

        # Увеличиваем счетчик итераций
        self.k += 1

        return weight_diff  # Возвращаем разницу весов

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента для функции потерь
        :param x: матрица признаков
        :param y: вектор целевых значений
        :return: градиент: np.ndarray
        """
        # Рассчитываем предсказания
        y_pred = self.predict(x)

        # Вычисляем ошибку (разницу между истинными значениями и предсказанными)
        error = y - y_pred

        # Вычисляем градиент: -2/N * X^T * error
        gradient = -2 / x.shape[0] * np.dot(x.T, error)

        return gradient

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Вычисление функции потерь (например, MSE)"""
        y_pred = self.predict(x)  # Получаем предсказания
        error = y - y_pred
        mse = np.mean(np.square(error))  # Среднеквадратичная ошибка (MSE)
        return mse


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: размерность данных
        :param lambda_: коэффициент регуляризации
        :param loss_function: функция потерь
        """
        # Вызов конструктора родительского класса для инициализации всех атрибутов
        super().__init__(dimension=dimension, lambda_=lambda_, loss_function=loss_function)
        
        # Устанавливаем параметр инерции alpha
        self.alpha = 0.9

        # Инициализация "скорости" (momentum) h
        self.h = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием метода инерции (momentum).
        :param gradient: градиент функции потерь
        :return: разница весов (w_{k + 1} - w_k): np.ndarray
        """
        # Обновление значения "скорости" (momentum)
        self.h = self.alpha * self.h + self.learning_rate * gradient

        # Обновление весов с использованием момента
        weight_diff = self.h  # Разница весов

        # Обновление весов
        self.w -= weight_diff

        return weight_diff

class Adam(VanillaGradientDescent):
    """
    Adam gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: размерность данных
        :param lambda_: коэффициент регуляризации
        :param beta1: коэффициент для 1-го момента
        :param beta2: коэффициент для 2-го момента
        :param epsilon: небольшое значение для предотвращения деления на ноль
        :param loss_function: функция потерь
        """
        # Вызов конструктора родительского класса для инициализации всех атрибутов
        super().__init__(dimension=dimension, lambda_=lambda_, loss_function=loss_function)
        
        # Параметры для алгоритма Adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Инициализация моментов
        self.m = np.zeros(dimension)  # Момент для первого порядка (градиенты)
        self.v = np.zeros(dimension)  # Момент для второго порядка (квадраты градиентов)

        # Счетчик шагов
        self.t = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием алгоритма Adam.
        :param gradient: градиент функции потерь
        :return: разница весов (w_{k + 1} - w_k): np.ndarray
        """
        # Увеличиваем счетчик шагов
        self.t += 1

        # Обновление моментов первого и второго порядка
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

        # Коррекция смещения моментов
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Вычисление шага
        weight_diff = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Обновление весов
        self.w -= weight_diff

        return weight_diff

def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    # Извлекаем параметры из конфигурации
    kwargs = descent_config.get('kwargs', {})

    # Проверяем, что 'dimension' является положительным целым числом
    dimension = kwargs.get('dimension', 0)  # Если 'dimension' отсутствует, по умолчанию 0

    if dimension <= 0:
        raise ValueError(f"Dimension must be a positive integer, got {dimension}")

    # Убедимся, что dimension — целое число
    kwargs['dimension'] = int(dimension)

    # Словарь для отображения имен спусков на соответствующие классы
    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    # Проверка на корректность названия метода спуска
    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {list(descent_mapping.keys())}')

    descent_class = descent_mapping[descent_name]

    # Создание и возврат объекта спуска с корректно переданными аргументами
    return descent_class(**kwargs)
