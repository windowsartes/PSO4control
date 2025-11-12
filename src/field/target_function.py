from abc import ABC, abstractmethod

from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    x: float
    y: float


class TargetFunctionInterface(ABC):
    @abstractmethod
    def __call__(
        self,
        point: Point,
    ) -> float:
        pass


TARGET_FUNCTION_REGISTER: dict[str, type[TargetFunctionInterface]] = {}


def target_function(
    function: type[TargetFunctionInterface],
) -> type[TargetFunctionInterface]:
    TARGET_FUNCTION_REGISTER[function.__name__.lower()] = function

    return function


@target_function
class Gaussian(TargetFunctionInterface):
    def __init__(
        self,
        centre: Point = Point(5, 5),
        sigma: float = 10,
    ):
        super().__init__()

        self._centre: Point = centre
        self._sigma: float = sigma

    def __call__(
        self,
        point: Point,
    ) -> float:
        return (1/(self._sigma*np.sqrt(2*np.pi))) * \
               np.exp(-((point.x - self._centre.x)**2 + (point.y - self._centre.y)**2)/(2*self._sigma))


@target_function
class Rastrigin(TargetFunctionInterface):
    def __init__(
        self,
        centre: Point = Point(5, 5),
        sigma: None = None,
    ):
        super().__init__()

        self._centre: Point = centre

    def __call__(
        self,
        point: Point,
    ) -> float:
        return -(20 + ((point.x - self._centre.x)**2 - 10 * np.cos(2 * np.pi * (point.x - self._centre.x))) + \
            ((point.y - self._centre.y)**2 - 10 * np.cos(2 * np.pi * (point.y - self._centre.y)))) + 90 + 1e-8
    

@target_function
class Griewank(TargetFunctionInterface):
    def __init__(
        self,
        centre: Point = Point(5, 5),
        sigma: None = None,
    ):
        super().__init__()

        self._centre: Point = centre

    def __call__(
        self,
        point: Point,
    ) -> float:
        return -(1 + ((point.x - self._centre.x)** 2 + (point.y - self._centre.y) ** 2) / 4000 - \
            np.cos((point.x - self._centre.x)) * np.cos((point.y - self._centre.y) / np.sqrt(2))) + 2.0125 + 1e-8
