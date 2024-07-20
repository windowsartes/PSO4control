import typing as tp

import numpy as np
import sympy
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


TARGET_FUNCTION_REGISTER: dict[str, tp.Callable[[Point, tp.Any], float]] = {}


def target_function(
    function: tp.Callable[[Point, tp.Any], float]
) -> tp.Callable[[Point, tp.Any], float]:
    TARGET_FUNCTION_REGISTER[function.__name__.lower()] = function
    return function


@dataclass
class SympyPoint:
    x: sympy.Symbol
    y: sympy.Symbol


TARGET_FUNCTION_SYMBOLIC_REGISTER: dict[str, tp.Callable[[SympyPoint, tp.Any], sympy.Expr]] = {}

def target_function_symbolic(
    function: tp.Callable[[SympyPoint, tp.Any], tp.Any],
) -> tp.Callable[[SympyPoint, tp.Any], sympy.Expr]:
    TARGET_FUNCTION_SYMBOLIC_REGISTER[function.__name__[:-9]] = function
    return function


@target_function
def gaussian(
    point: Point,
    centre: Point = Point(5, 5),
    sigma: float = 10,
) -> float:
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((point.x - centre.x)**2 + (point.y - centre.y)**2)/(2*sigma))


@target_function_symbolic
def gaussian_symbolic(
    point: SympyPoint,
    centre: Point = Point(5, 5),
    sigma: float = 10,
) -> sympy.Expr:
    return (1/(sigma*np.sqrt(2*np.pi))) * sympy.exp(-((point.x - centre.x)**2 + (point.y - centre.y)**2)/(2*sigma))
