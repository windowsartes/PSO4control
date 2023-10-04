import pickle
import typing as tp
from abc import ABC, abstractmethod

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import uniform

from src.solvers.solver_interface import SolverInterface


class GradientMethodInterface(SolverInterface):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass

    @position.setter
    @abstractmethod
    def position(self, new_position: np.ndarray[tp.Any, np.dtype[np.float64]]) -> None:
        pass

    @property
    @abstractmethod
    def total_path_length(self) -> float:
        pass

    @abstractmethod
    def _correct_position(self) -> None:
        pass

    @abstractmethod
    def run(self) -> tuple[tp.Any, ...]:
        pass

    @abstractmethod
    def show_current_position(self, title: str) -> None:
        pass


class GradientMethodBase(GradientMethodInterface):
    def __init__(self, n_iterations: int, scene):  # type: ignore
        self._scene = scene
        self._score: float = float("-inf")
        self._n_iterations: int = n_iterations

        self._total_path_length: float = 0

        edge: int = np.random.randint(4)
        self._position: np.ndarray[tp.Any, np.dtype[np.float64]]
        if edge == 0:  # left
            self._position = np.array([0, uniform(0, self._scene.field.height)])
        elif edge == 1:  # right
            self._position = np.array([self._scene.field.width, uniform(0, self._scene.field.height)])
        elif edge == 2:  # top
            self._position = np.array([uniform(0, self._scene.field.width), 0])
        else:  # bottom
            self._position = np.array([uniform(0, self._scene.field.width), self._scene.field.height])

        if self._scene.verbosity.value > 1:
            self.show_current_position("Начальное положение")

    @property
    def position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._position

    @position.setter
    def position(self, new_position: np.ndarray[tp.Any, np.dtype[np.float64]]) -> None:
        self._position = new_position
        self._correct_position()

    @property
    def total_path_length(self) -> float:
        return self._total_path_length

    def _correct_position(self) -> None:
        if self._position[0] < 0:
            self._position[0] = 0
        elif self._position[0] > self._scene.field.width:
            self._position[0] = self._scene.field.width

        if self._position[1] < 0:
            self._position[1] = 0
        elif self._position[1] > self._scene.field.height:
            self._position[1] = self._scene.field.height

    def show_current_position(self, title: str) -> None:
        correctness_scale = 100

        figure = pickle.load(open("./stored_field/field.pickle", "rb"))
        ax = plt.gca()

        x, y = 100, 100

        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            figure.canvas.manager.window.move(x, y)

        ax.scatter(self.position[0] * correctness_scale, self.position[1] * correctness_scale,
                   marker='o', color='b', ls='', s=20)

        ax.set_xlim(0, self._scene.field.width * correctness_scale)
        ax.set_ylim(0, self._scene.field.height * correctness_scale)
        ax.set_title(title)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.5)
        plt.close(figure)


class GradientLift(GradientMethodBase):
    def __init__(self, n_iterations: int, scene) -> None:  # type: ignore
        super().__init__(n_iterations, scene)

    def run(self) -> tuple[int | float, ...]:
        for i in range(1, self._n_iterations + 1):
            current_gradient: np.ndarray[tp.Any, np.dtype[np.float64]] = self._scene.field.gradient(*self._position)

            current_norm = np.linalg.norm(current_gradient)
            if current_norm < self._scene.hyperparameters.early_stopping_epsilon:
                break

            if current_norm >= (self._scene.field.height / self._scene.hyperparameters.velocity_factor):
                current_gradient = (current_gradient / current_norm) * \
                                   (self._scene.field.height / self._scene.hyperparameters.velocity_factor)

            self._position = self._position + current_gradient
            self._correct_position()
            self._total_path_length += float(np.linalg.norm(current_gradient))
            self._score = self._scene.field.target_function(*self._position)

            if self._scene.verbosity.value > 0:
                if i % self._scene.verbosity.show_period == 0:
                    self.show_current_position(str(i))

        return (i,
                self._scene.answer.value - self._score,
                (self._scene.answer.value - self._score)/self._scene.answer.value,
                self._total_path_length)


class NewtonMethod(GradientMethodBase):
    def __init__(self, n_iterations: int, scene):  # type: ignore
        super().__init__(n_iterations, scene)

    def run(self) -> tuple[int | float, ...]:
        for i in range(1, self._n_iterations + 1):
            current_gradient: np.ndarray[tp.Any, np.dtype[np.float64]] = self._scene.field.gradient(*self._position)
            hessian_inv = np.linalg.inv(self._scene.field.hessian(*self._position))

            current_shift = hessian_inv @ current_gradient

            current_norm = np.linalg.norm(current_shift)
            if current_norm < self._scene.hyperparameters.early_stopping_epsilon:
                break

            if current_norm >= (self._scene.field.height / self._scene.hyperparameters.velocity_factor):
                current_shift = (current_shift/current_norm) * \
                                (self._scene.field.height / self._scene.hyperparameters.velocity_factor)

            self._position = self._position - current_shift
            self._correct_position()

            self._total_path_length += float(np.linalg.norm(current_gradient))
            self._score = self._scene.field.target_function(*self._position)

            if self._scene.verbosity.value > 0:
                if i % self._scene.verbosity.show_period == 0:
                    self.show_current_position(str(i))

        return (i,
                self._scene.answer.value - self._score,
                (self._scene.answer.value - self._score)/self._scene.answer.value,
                self._total_path_length)
