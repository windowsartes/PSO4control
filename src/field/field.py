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
    centre: tuple[tuple[float, float], ...]
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
        target_function: type[tf.TargetFunctionInterface],
        # target_function_symbolic: tp.Callable[[tf.SympyPoint], sympy.Expr],
    ):
        self._parameters: FieldParameters = parameters
        self._additional_parameters: tp.Optional[AdditionalParameter] = additional_parameters
        self._target_function: tf.TargetFunctionInterface = target_function(  # type: ignore
            tf.Point(*self._parameters.centre),
            self._parameters.sigma,
        )
        self._target_max_value: float = self._target_function(tf.Point(*self._parameters.centre))

        if self._additional_parameters is not None:
            self._additional_target_functions: list[tf.TargetFunctionInterface] = []
            self._additional_max_values: list[float] = []

            for i in range(len(self._additional_parameters.centre)):
                self._additional_target_functions.append(
                    target_function(  # type: ignore
                        tf.Point(*self._additional_parameters.centre[i]),
                        self._additional_parameters.sigma,
                    )
                )
                self._additional_max_values.append(
                    self._additional_target_functions[-1](tf.Point(*self._additional_parameters.centre[i]))
                )

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
            return 0.

        target_function_value: float = self._target_function(tf.Point(x, y))

        for i in range(len(self._additional_target_functions)):
            candidate: float = \
                (self._additional_target_functions[i](tf.Point(x, y)) / self._additional_max_values[i]) * \
                (self._target_max_value*self._additional_parameters.coeff)

            if candidate > target_function_value:
                return 1.

        return 0.

    def target_function(
        self,
        x: float,
        y: float,
    ) -> float:
        if self._additional_parameters is None:
            return self._target_function(tf.Point(x, y))

        max_value: float = self._target_function(tf.Point(x, y))

        for i in range(len(self._additional_target_functions)):
            candidate: float = \
                (self._additional_target_functions[i](tf.Point(x, y)) / self._additional_max_values[i]) * \
                (self._target_max_value*self._additional_parameters.coeff)

            max_value = max(max_value, candidate)

        return max_value

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
