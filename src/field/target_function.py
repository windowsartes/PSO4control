from abc import ABC, abstractmethod

from dataclasses import dataclass

import numpy as np
# import sympy


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
        value: float = (1/(self._sigma*np.sqrt(2*np.pi))) * \
                       np.exp(-((point.x - self._centre.x)**2 + (point.y - self._centre.y)**2)/(2*self._sigma))
        return value


"""
@dataclass
class SympyPoint:
    x: sympy.Symbol
    y: sympy.Symbol


TARGET_FUNCTION_SYMBOLIC_REGISTER: dict[str, tp.Callable[[SympyPoint], sympy.Expr]] = {}


def target_function_symbolic(
    function: tp.Callable[[SympyPoint], sympy.Expr],
) -> tp.Callable[[SympyPoint], sympy.Expr]:
    TARGET_FUNCTION_SYMBOLIC_REGISTER[function.__name__[:-9]] = function
    return function


@target_function_symbolic
def gaussian_symbolic(
    point: SympyPoint,
    centre: Point = Point(5, 5),
    sigma: float = 10,
) -> sympy.Expr:
    return (1/(sigma*np.sqrt(2*np.pi))) * sympy.exp(-((point.x - centre.x)**2 + (point.y - centre.y)**2)/(2*sigma))
"""
