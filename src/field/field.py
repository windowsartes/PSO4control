import pathlib
import pickle
import typing as tp
from abc import ABC, abstractmethod
from math import ceil

import numpy as np
import seaborn as sns
import sympy
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from pydantic import BaseModel

from src.field import target_function as tf


class FieldParameters(BaseModel):
    height: float
    width: float
    quality_scale: float

class FieldInterface(ABC):
    """
    This interface can be used as a template for your filed. It must contain float height abd width attributes,
    and 2 functions: target_function, which describes your field at its every point, and target_function_symbolic, which
    is sympy version on target_function = it will be used if you want to compute gradient ot hessian. Sometimes you can
    use only 1 function for these attributes, but sometimes it's impossible. For example, np.exp can't work with
    sympy.Symbol, meanwhile sympy.exp is too slow, so it's very inefficient to use it during regular computing.
    """

    @property
    @abstractmethod
    def height(self) -> float:
        """
        This method will return field's height;

        Returns: field's height
        """
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        """
        This method will return field's width;

        Returns: field's width
        """
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

    @abstractmethod
    def gradient(
        self,
        x: float,
        y: float,
    ) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        """
        Computes and returns field's gradient in the given point using symbolic version of a target function;
        Args:
            x: x coordinate of a point of interest;
            y: y coordinate of a point of interest;

        Returns: field's gradient in the given point;
        """
        pass

    @abstractmethod
    def show(self) -> None:
        """
        Creates a grid and computes field's value at its every point, then shows it in both 2D and 3D versions;

        Returns: None;
        """
        pass

    @abstractmethod
    def compute_and_save_field(
        self,
        path_to_file: str | pathlib.Path,
    ) -> None:
        """
        Creates a grid and computes field's value at its every point, then saves this field as a pickle object by
        the given path;
        Args:
            path_to_file: path where you want to save the field, so you can use it later;
        Returns: Nothing;
        """
        pass

class Field(FieldInterface):
    def __init__(
        self,
        parameters: FieldParameters,
        target_function: tp.Callable[[tf.Point, tp.Any], float],
        target_function_symbolic: tp.Callable[[tf.SympyPoint, tp.Any], sympy.Expr],
    ):
        self._parameters: float = parameters
        self._target_function: tp.Callable[[tf.Point, tp.Any], float] = target_function
        self._target_function_symbolic: tp.Callable[[tf.SympyPoint, tp.Any], sympy.Expr] = \
            target_function_symbolic(tf.SympyPoint(sympy.Symbol('x'), sympy.Symbol('y'))) 
        self._gradient: sympy.Expr = sympy.Matrix([self._target_function_symbolic]).jacobian(
            sympy.Matrix([sympy.Symbol('x'), sympy.Symbol('y')])
        )
        self._hessian: sympy.Expr = sympy.hessian(
            self._target_function_symbolic,
            [sympy.Symbol('x'), sympy.Symbol('y')]
        )

    @property
    def height(self) -> float:
        return self._parameters.height

    @property
    def width(self) -> float:
        return self._parameters.width

    @property
    def quality_scale(self) -> float:
        return self._parameters.quality_scale

    def target_function(
        self,
        x: float,
        y: float,
    ) -> float:
        return self._target_function(tf.Point(x, y))

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
        """
        Computes and return you a hessian in given point as s (2,2)-matrix since it's 2d field;
        Args:
            x: x coordinate of a point of interest;
            y: y coordinate of a point of interest;

        Returns: a hessian in given point as a (2,2) np.ndarray;

        """
        return np.array([float(value) for value in self._hessian.subs([('x', x), ('y', y)])]).reshape((2, 2))

    def show(self) -> None:
        x_values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.linspace(0, self._parameters.width, ceil(self._parameters.width * self._parameters.quality_scale))
        y_values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.linspace(0, self._parameters.height, ceil(self._parameters.height * self._parameters.quality_scale))

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = np.stack((x_grid.flatten(), y_grid.flatten()), -1)

        values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.array([self._target_function(tf.Point(x, y)) for x, y in coordinates])  # type: ignore

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
            np.linspace(0, self._parameters.width, ceil(self._parameters.width * self._parameters.quality_scale))
        y_values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.linspace(0, self._parameters.height, ceil(self._parameters.height * self._parameters.quality_scale))

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = np.stack((x_grid.flatten(), y_grid.flatten()), -1)

        values: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.array([self._target_function(tf.Point(x, y)) for x, y in coordinates])
        values = values.reshape((len(x_values), len(y_values)))
        sns.heatmap(values, cmap=cm["hot"])
        plt.axis('off')

        with open(path_to_file, "wb") as f:
            pickle.dump(figure, f)

        plt.close()
