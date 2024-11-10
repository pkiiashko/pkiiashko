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

        # Если dimension == 0, вызываем метод init_weights для инициализации весов
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
        # Явно передаем dimension и остальные параметры в базовый класс через kwargs
        super().__init__(dimension=dimension, **kwargs)  # Инициализируем базовый класс

        self.learning_rate = learning_rate  # Начальная скорость обучения
        self.lambda_ = lambda_  # Коэффициент регуляризации
        self.s0 = s0  # Параметр s0 для вычисления eta
        self.p = p  # Параметр p для вычисления eta
        self.loss_function = loss_function  # Функция потерь
        self.k = 0  # Счетчик итераций (для вычисления eta)

        # Инициализируем веса, если они еще не были заданы
        if self.w is None or len(self.w) == 0:
            self.init_weights(self.dimension)  # Инициализация весов нулями

    def init_weights(self, dimension: int) -> None:
        """Инициализация весов нулями."""
        if dimension > 0:
            self.w = np.zeros(dimension)  # Инициализируем веса нулями

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновить веса с учетом градиента
        :param gradient: градиент функции потерь
        :return: разница весов (w_{k + 1} - w_k): np.ndarray
        """
        # Вычисляем длину шага по формуле
        eta_k = self.lambda_ / (self.s0 + self.k) ** self.p

        # Вычисляем разницу весов (w_{k + 1} - w_k) = -eta_k * gradient
        weight_diff = -eta_k * gradient
        
        # Обновляем веса
        self.w -= weight_diff

        # Увеличиваем счетчик итераций
        self.k += 1

        return weight_diff

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



class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE, **kwargs):
        """
        :param batch_size: batch size (int)
        :param dimension: размерность данных
        :param lambda_: коэффициент регуляризации
        :param loss_function: функция потерь
        """
        # Вызов конструктора родительского класса для инициализации весов и других параметров
        super().__init__(dimension=dimension, lambda_=lambda_, loss_function=loss_function, **kwargs)
        self.batch_size = batch_size  # Размер батча для стохастического градиентного спуска

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента для стохастического градиентного спуска
        :param x: матрица признаков
        :param y: вектор целевых значений
        :return: градиент: np.ndarray
        """
        # Сначала случайным образом выбираем индексы для мини-батча
        N = x.shape[0]  # Количество примеров
        batch_indices = np.random.choice(N, self.batch_size, replace=False)

        # Получаем подмножество данных (мини-батч)
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]

        # Рассчитываем предсказания для этого батча
        y_pred = self.predict(x_batch)

        # Ошибка для этого батча
        error = y_batch - y_pred

        # Вычисляем градиент для MSE с L2-регуляризацией
        gradient = -2 / self.batch_size * np.dot(x_batch.T, error) + 2 * self.lambda_ * self.w

        return gradient

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновить веса с учетом градиента
        :param gradient: градиент функции потерь
        :return: разница весов (w_{k + 1} - w_k): np.ndarray
        """
        # Вычисляем длину шага по формуле
        eta_k = self.lambda_ / (self.s0 + self.k) ** self.p

        # Вычисляем разницу весов (w_{k + 1} - w_k) = -eta_k * gradient
        weight_diff = -eta_k * gradient
        
        # Обновляем веса
        self.w -= weight_diff

        # Увеличиваем счетчик итераций
        self.k += 1

        return weight_diff


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
    Adam gradient descent with momentum and adaptive learning rate
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.epsilon: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m / (1 - self.beta1 ** (self.k + 1))
        v_hat = self.v / (1 - self.beta2 ** (self.k + 1))

        weight_diff = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

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
