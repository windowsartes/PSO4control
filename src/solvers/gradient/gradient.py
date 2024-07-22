import pickle
import typing as tp
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use('TKAgg')
import numpy as np
from matplotlib import pyplot as plt

from src.solvers.solver_interface import SolverInterface
from src.solvers.gradient import gradient_params


class GradientMethodInterface(SolverInterface):
    @abstractmethod
    def turn(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def show(
        self,
        title: str,
    ) -> None:
        pass


class GradientMethodBase(GradientMethodInterface):
    def __init__(
        self,
        params: gradient_params.GradientParams,
        field_size: float,
        field_quality_scale: float,
    ):
        self._field_size: float = field_size
        self._field_quality_scale: float = field_quality_scale

        self._position: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((2,))
        self._velocity: np.ndarray[tp.Any, np.dtype[np.float64]] = np.zeros((2,))

        self._path_length: float = 0.0

        self._velocity_factor: float = params.velocity_factor

        edge: int = np.random.randint(4)
        if edge == 0:  # left
            self._position = np.array(
                [
                    0,
                    np.random.uniform(0, field_size),
                ]
            )
        elif edge == 1:  # right
            self._position = np.array(
                [
                    field_size,
                    np.random.uniform(0, field_size),
                ]
            )
        elif edge == 2:  # top
            self._position = np.array(
                [
                    np.random.uniform(0, field_size),
                    0,
                ]
            )
        else:  # bottom
            self._position = np.array(
                [
                    np.random.uniform(0, field_size),
                    field_size,
                ]
            )

    @property
    def position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._position

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def velocity(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._velocity

    def show(
        self,
        title: str,
    ) -> None:
        with open("./stored_field/field.pickle", "rb") as f:
            figure = pickle.load(f)
        ax = plt.gca()

        x, y = 100, 100

        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))

        ax.scatter(
            self.position[0] * self._field_quality_scale,
            self.position[1] * self._field_quality_scale,
            marker='o',
            color='b',
            ls='',
            s=20,
        )

        ax.set_xlim(0, self._field_size * self._field_quality_scale)
        ax.set_ylim(0, self._field_size * self._field_quality_scale)
        ax.set_title(title)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.5)
        plt.close(figure)


SOLVER_REGISTER: dict[str, tp.Type[GradientMethodBase]] = {}

def solver(
    cls: tp.Type[GradientMethodBase],
) -> tp.Type[GradientMethodBase]:
    SOLVER_REGISTER[cls.__name__.lower()] = cls
    return cls


@solver
class GradientLift(GradientMethodBase):
    def __init__(
        self,
        params: gradient_params.GradientParams,
        field_size: float,
        feild_quality_scale: float,
    ):
        super().__init__(params, field_size, feild_quality_scale)

    def turn(
        self,
        gradient: np.ndarray[tp.Any, np.dtype[np.float64]],
    ):
        self._velocity = gradient

        velocity_norm: float = np.linalg.norm(self._velocity)

        if velocity_norm > self._field_size / self._velocity_factor:
            self._velocity = (self._velocity / velocity_norm) * (self._field_size / self._velocity_factor)

        self._position = self._position + self._velocity
        self._path_length += float(np.linalg.norm(self._velocity))

@solver
class NewtonsMethod(GradientMethodBase):
    def __init__(
        self,
        params: gradient_params.GradientParams,
        field_size: float,
        feild_quality_scale: float,
    ):
        super().__init__(params, field_size, feild_quality_scale)

    def turn(
        self,
        gradient: np.ndarray[tp.Any, np.dtype[np.float64]],
        hessian:  np.ndarray[tp.Any, np.dtype[np.float64]],
    ):
        self._velocity = np.linalg.inv(hessian) @ gradient

        velocity_norm: float = np.linalg.norm(self._velocity)

        if velocity_norm > self._field_size / self._velocity_factor:
            self._velocity = (self._velocity / velocity_norm) * (self._field_size / self._velocity_factor)

        self._position = self._position + self._velocity
        self._path_length += float(np.linalg.norm(self._velocity))
