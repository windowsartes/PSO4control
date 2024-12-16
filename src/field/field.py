import pathlib
import pickle
import typing as tp
from abc import ABC, abstractmethod
from math import ceil

import numpy as np
import seaborn as sns
# import sympy
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from pydantic import BaseModel

from src.field import target_function as tf


class FieldParameters(BaseModel):
    size: float
    quality_scale: float
    centre: tuple[float, float]
    sigma: float


class AdditionalParameter(BaseModel):
    centre: tuple[float, float]
    sigma: float
    coeff: float


class FieldInterface(ABC):
    @property
    @abstractmethod
    def size(self) -> float:
        pass

    @property
    @abstractmethod
    def quality_scale(self) -> float:
        pass

    @abstractmethod
    def target_function(
        self,
        x: float,
        y: float,
    ) -> float:
        pass

    """
    @abstractmethod
    def gradient(
        self,
        x: float,
        y: float,
    ) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def hessian(
        self,
        x: float,
        y: float,
    ) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass
    """

    @abstractmethod
    def show(self) -> None:
        pass

    @abstractmethod
    def compute_and_save_field(
        self,
        path_to_file: str | pathlib.Path,
    ) -> None:
        pass


class Field(FieldInterface):
    def __init__(
        self,
        parameters: FieldParameters,
        additional_parameters: tp.Optional[AdditionalParameter],
        target_function: tp.Type[tp.Callable[[tf.Point], float]],
        # target_function_symbolic: tp.Callable[[tf.SympyPoint], sympy.Expr],
    ):
        self._parameters: FieldParameters = parameters
        self._additional_parameters: tp.Optional[AdditionalParameter] = additional_parameters
        self._target_function: tp.Callable[[tf.Point], float] = target_function(  # type: ignore
            tf.Point(*self._parameters.centre),
            self._parameters.sigma,
        )
        self._target_max_value = self._target_function(tf.Point(*self._parameters.centre))
        if self._additional_parameters is not None:
            self._additional_target_function: tp.Callable[[tf.Point], float] = target_function(  # type: ignore
                tf.Point(*self._additional_parameters.centre),
                self._additional_parameters.sigma,
            )
            self._additional_max_value: float = \
                self._additional_target_function(tf.Point(*self._additional_parameters.centre))
        """
        self._target_function_symbolic: sympy.Expr = \
            target_function_symbolic(tf.SympyPoint(sympy.Symbol('x'), sympy.Symbol('y')))
        self._gradient: sympy.Expr = sympy.Matrix([self._target_function_symbolic]).jacobian(
            sympy.Matrix([sympy.Symbol('x'), sympy.Symbol('y')])
        )
        self._hessian: sympy.Expr = sympy.hessian(
            self._target_function_symbolic,
            [sympy.Symbol('x'), sympy.Symbol('y')]
        )
        """

    @property
    def size(self) -> float:
        return self._parameters.size

    @property
    def quality_scale(self) -> float:
        return self._parameters.quality_scale

    def check_additional(
        self,
        x: float,
        y: float,
    ) -> float:
        if self._additional_parameters is None:
            return self._target_function(tf.Point(x, y))

        target_function_value = self._target_function(tf.Point(x, y))
        additional_function_value = (self._additional_target_function(tf.Point(x, y)) / self._additional_max_value) * \
            (self._target_max_value*self._additional_parameters.coeff)

        return 1. if additional_function_value > target_function_value else 0.

    def target_function(
        self,
        x: float,
        y: float,
    ) -> float:
        if self._additional_parameters is None:
            return self._target_function(tf.Point(x, y))

        target_function_value = self._target_function(tf.Point(x, y))
        additional_function_value = (self._additional_target_function(tf.Point(x, y)) / self._additional_max_value) * \
            (self._target_max_value*self._additional_parameters.coeff)

        additional_is_bigger = 1. if additional_function_value > target_function_value else 0.

        return max(target_function_value, additional_function_value)

        '''
        return max(
            self._target_function(tf.Point(x, y)),
            (self._additional_target_function(tf.Point(x, y)) / self._additional_max_value) *
            (self._target_max_value*self._additional_parameters.coeff),
        )
        '''

    """
    def gradient(
        self,
        x: float,
        y: float,
    ) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return np.array([float(value) for value in self._gradient.subs([('x', x), ('y', y)])])

    def hessian(
        self,
        x: float,
        y: float,
    ) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return np.array([float(value) for value in self._hessian.subs([('x', x), ('y', y)])]).reshape((2, 2))
    """

    def show(self) -> None:
        x_values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.linspace(0, self._parameters.size, ceil(self._parameters.size * self._parameters.quality_scale))
        y_values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.linspace(0, self._parameters.size, ceil(self._parameters.size * self._parameters.quality_scale))

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = np.stack((x_grid.flatten(), y_grid.flatten()), -1)

        values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.array([self.target_function(x, y) for x, y in coordinates])
        values = values.reshape((len(x_values), len(y_values)))

        sns.heatmap(data=values, cmap=cm["hot"])
        plt.axis('off')

        figure, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            x_grid,
            y_grid,
            values,
            cmap=cm["hot"],
            linewidth=0,
            antialiased=False,
        )
        figure.colorbar(surf, ax=ax, shrink=0.5, aspect=15)

        plt.show()

    def compute_and_save_field(
        self,
        path_to_file: str | pathlib.Path,
    ) -> None:
        figure, ax = plt.subplots()

        x_values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.linspace(0, self._parameters.size, ceil(self._parameters.size * self._parameters.quality_scale))
        y_values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.linspace(0, self._parameters.size, ceil(self._parameters.size * self._parameters.quality_scale))

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = np.stack((x_grid.flatten(), y_grid.flatten()), -1)

        values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.array([self.target_function(x, y) for x, y in coordinates])
        values = values.reshape((len(x_values), len(y_values)))

        sns.heatmap(values, cmap=cm["hot"])
        plt.axis('off')

        with open(path_to_file, "wb") as f:
            pickle.dump(figure, f)

        plt.close()
