import pickle
import typing as tp
from abc import ABC, abstractmethod
from math import ceil

import numpy as np
import seaborn as sns
import sympy
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm


def gaussian(x: float, y: float, centre: tp.Optional[tuple[float, float]] = (5, 5),
             sigma: tp.Optional[float] = 10) -> float:
    x0: float = centre[0]
    y0: float = centre[1]

    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma))


def gaussian_symbolic(x: sympy.Symbol, y: sympy.Symbol, centre: tp.Optional[tuple[float, float]] = (5, 5),
                      sigma: tp.Optional[float] = 10) -> tp.Any:
    x0: float = centre[0]
    y0: float = centre[1]

    return (1/(sigma*np.sqrt(2*np.pi))) * sympy.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma))


class FieldInterface(ABC):
    """
    This interface can be used as a template for your filed. It must contain float height abd width attributes,
    and 2 functions: target_function, which describes your field at its every point, and target_function_symbolic, which
    is sympy version on target_function = it will be used if you want to compute gradient ot hessian. Sometimes you can
    use only 1 function for these attributes, but sometimes it's impossible. For example, np.exp can't work with
    sympy.Symbol, meanwhile sympy.exp is too slow, so it's very inefficient to use it during regular computing.
    """
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def height(self):
        """
        This method will return field's height;

        Returns: field's height
        """
        pass

    @property
    @abstractmethod
    def width(self):
        """
        This method will return field's width;

        Returns: field's width
        """
        pass

    @property
    @abstractmethod
    def quality_scale(self):
        pass

    @abstractmethod
    def target_function(self):
        """
        Returns field's target function so there is no need to call a class attribute outside the class;

        Returns: field's target function
        """
        pass

    @abstractmethod
    def gradient(self, x: float, y: float):
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

        Returns: Nothing;
        """
        pass

    @abstractmethod
    def compute_and_save_field(self, path_to_file: tp.Optional[str]):
        """
        Creates a grid and computes field's value at its every point, then saves this field as a pickle object by
        the given path;
        Args:
            path_to_file: path where you want to save the field, so you can use it later;
        Returns: Nothing;
        """
        pass


class GaussianField(FieldInterface):
    def __init__(self, height: float, width: float, quality_scale: float,
                 target_function: tp.Callable[[float, float, tp.Optional[tuple[float, float]],
                                               tp.Optional[float]], float],
                 target_function_symbolic: tp.Callable[[sympy.Symbol, sympy.Symbol, tp.Optional[tuple[float, float]],
                                                        tp.Optional[float]], tp.Any]):
        self._height: float = height
        self._width: float = width
        self._quality_scale: float = quality_scale
        self._target_function: tp.Callable[[float, float, tp.Optional[tuple[float, float]],
                                            tp.Optional[float]], float] = target_function
        self._f: tp.Callable[[sympy.Symbol, sympy.Symbol, tp.Optional[tuple[float, float]],
                              tp.Optional[float]], tp.Any] = \
            target_function_symbolic(sympy.Symbol('x'), sympy.Symbol('y'))
        self._gradient = sympy.Matrix([self._f]).jacobian(sympy.Matrix([sympy.Symbol('x'), sympy.Symbol('y')]))

        self._hessian = sympy.hessian(self._f, [sympy.Symbol('x'), sympy.Symbol('y')])

    @property
    def height(self) -> float:
        return self._height

    @property
    def width(self) -> float:
        return self._width

    @property
    def target_function(self) -> tp.Callable[[float, float, tp.Optional[tuple[float, float]],
                                              tp.Optional[float]], float]:
        return self._target_function

    def gradient(self, x: float, y: float) -> np.ndarray:
        return np.array([float(value) for value in self._gradient.subs([('x', x), ('y', y)])])

    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        Computes and return you a hessian in given point as s (2,2)-matrix since it's 2d field;
        Args:
            x: x coordinate of a point of interest;
            y: y coordinate of a point of interest;

        Returns: a hessian in given point as a (2,2) np.ndarray;

        """
        return np.array([float(value) for value in self._hessian.subs([('x', x), ('y', y)])]).reshape((2, 2))

    def show(self, field_size_scale: int = 100) -> None:
        # since width and height are round, ceil doesn't affect;
        x_values: np.ndarray = np.linspace(0, self._width, ceil(self._width * field_size_scale))
        y_values: np.ndarray = np.linspace(0, self._height, ceil(self._height * field_size_scale))

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        coordinates = np.stack((x_grid.flatten(), y_grid.flatten()), -1)

        values = np.array([self._target_function(x, y) for x, y in coordinates])

        values = values.reshape((len(x_values), len(y_values)))

        sns.heatmap(data=values, cmap=cm["hot"])
        plt.axis('off')

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x_grid, y_grid, values, cmap=cm["hot"],
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15)

        plt.show()

    def compute_and_save_field(self, path_to_file: str = None, field_size_scale: int = 100) -> None:
        fig, ax = plt.subplots()

        x_values: np.ndarray = np.linspace(0, self._width, ceil(self._width * field_size_scale))
        y_values: np.ndarray = np.linspace(0, self._height, ceil(self._height * field_size_scale))

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        coordinates = np.stack((x_grid.flatten(), y_grid.flatten()), -1)

        values = np.array([self._target_function(x, y) for x, y in coordinates])
        values = values.reshape((len(x_values), len(y_values)))
        sns.heatmap(values, cmap=cm["hot"])
        plt.axis('off')

        if path_to_file is not None:
            pickle.dump(fig, open(path_to_file, "wb"))
        else:
            pickle.dump(fig, open("./stored_field/field.pickle", "wb"))

        plt.close()


if __name__ == "__main__":
    field = GaussianField(10, 10, gaussian, gaussian_symbolic)
    field.show()
