import pickle
import random
import typing as tp
from abc import abstractmethod

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.answer.answer import Point
from src.solvers.solver_interface import SolverInterface
from src.solvers.swarm.particle import Particle
from src.solvers.swarm import swarm_params

# matplotlib.use('TKAgg')


class SwarmInterface(SolverInterface):
    @abstractmethod
    def update_scores(
        self,
        particles_scores: list[float],
    ) -> None:
        pass

    @abstractmethod
    def correct_positions(
        self,
        field_size: float,
    ) -> None:
        pass

    @abstractmethod
    def turn(self) -> None:
        pass

    @abstractmethod
    def get_swarm_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def show(
        self,
        title: str,
    ) -> None:
        pass


class SwarmBase(SwarmInterface):
    def __init__(
        self,
        params: swarm_params.SwarmCentralizedParams,
        field_size: float,
        field_quality_scale: float,
    ) -> None:
        self._particles: list[Particle]

    def get_swarm_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        positions: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((len(self._particles), 2), dtype=np.double)

        for index, particle in enumerate(self._particles):
            positions[index] = particle.position

        return positions

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    def correct_positions(
        self,
        field_size: float,
    ) -> None:
        for i in range(len(self._particles)):
            self._particles[i].position[0] = max(self._particles[i].position[0], 0)
            self._particles[i].position[0] = min(self._particles[i].position[0], field_size)

            self._particles[i].position[1] = max(self._particles[i].position[1], 0)
            self._particles[i].position[1] = min(self._particles[i].position[1], field_size)

    def get_position_error(
        self,
        answer_point: Point,
        field_size: float,
    ) -> float:
        position_errors = []
        for particle in self._particles:
            position_errors.append(
                np.linalg.norm(
                    particle.best_position - np.array((answer_point.x, answer_point.y))
                ) / field_size
            )

        return float(np.mean(position_errors))

    def get_path_length(
        self,
    ) -> float:
        return sum([p.path_length for p in self._particles])


SOLVER_REGISTER: dict[str, tp.Type[SwarmBase]] = {}


def solver(
    cls: tp.Type[SwarmBase],
) -> tp.Type[SwarmBase]:
    SOLVER_REGISTER[cls.__name__[5:].lower()] = cls
    return cls


@solver
class SwarmCentralized(SwarmBase):
    def __init__(
        self,
        params: swarm_params.SwarmCentralizedParams,
        field_size: float,
        field_quality_scale: float,
    ):
        self._particles: list[Particle] = []
        for i in range(params.n_particles):
            self._particles.append(
                Particle(
                    field_size,
                    params.spawn,
                    params.coefficients,
                )
            )

        self._best_global_score: float = 0.
        self._best_global_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.empty((2))

        self._field_size: float = field_size
        self._field_quality_scale: float = field_quality_scale

    def update_scores(
        self,
        particles_scores: list[float],
    ) -> None:
        for i in range(len(self._particles)):
            if particles_scores[i] > self._particles[i].best_score:
                self._particles[i].best_score = particles_scores[i]
                self._particles[i]._best_position = self._particles[i].position

            if particles_scores[i] > self._best_global_score:
                self._best_global_score = particles_scores[i]
                self._best_global_position = self._particles[i].position

    def turn(self) -> None:
        for i in range(len(self._particles)):
            self._particles[i].move(self._best_global_position, self._field_size)

    def show(
        self,
        title: str,
    ) -> None:
        backend = matplotlib.get_backend()

        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = self.get_swarm_positions()

        with open("./stored_field/field.pickle", "rb") as f:
            figure = pickle.load(f)
        ax = plt.gca()

        x, y = 100, 100

        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))

        ax.scatter(
            coordinates[:, 0] * self._field_quality_scale,
            coordinates[:, 1] * self._field_quality_scale,
            marker='o',
            color='b',
            ls='',
            s=40,
        )

        ax.set_xlim(0, self._field_size * self._field_quality_scale)
        ax.set_ylim(0, self._field_size * self._field_quality_scale)
        ax.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            ax.annotate(
                label,
                (
                    coordinates[index][0] * self._field_quality_scale,
                    coordinates[index][1] * self._field_quality_scale,
                ),
                fontsize=10,
            )

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.5)
        plt.close(figure)


@solver
class SwarmDecentralized(SwarmBase):
    def __init__(
        self,
        params: swarm_params.SwarmDecentralizedParams,
        field_size: float,
        field_quality_scale: float,
    ):
        self._particles: list[Particle] = []
        for _ in range(params.n_particles):
            self._particles.append(
                Particle(
                    field_size,
                    params.spawn,
                    params.coefficients,
                )
            )

        self._best_global_scores: list[float] = [0.0] * params.n_particles
        self._best_global_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_particles

        self._field_size: float = field_size
        self._field_quality_scale: float = field_quality_scale

        self._connection_radius: float = params.connection_radius
        self._connection_dropout_probability: float = params.connection_dropout_probability

    def update_scores(
        self,
        particles_scores: list[float],
    ) -> None:
        for i in range(len(self._particles)):
            if particles_scores[i] > self._particles[i].best_score:
                self._particles[i].best_score = particles_scores[i]
                self._particles[i]._best_position = self._particles[i].position

        for i in range(len(self._particles)):
            for j in range(len(self._particles)):
                if all([
                    random.uniform(0, 1) > self._connection_dropout_probability,
                    np.linalg.norm(self._particles[i].position - self._particles[j].position) <
                        self._connection_radius * self._field_size,
                ]):
                    if self._best_global_scores[i] < self._particles[j].best_score:
                        self._best_global_scores[i] = self._particles[j].best_score
                        self._best_global_positions[i] = self._particles[j].best_position

    def turn(self) -> None:
        for i in range(len(self._particles)):
            self._particles[i].move(self._best_global_positions[i], self._field_size)

    def show(
        self,
        title: str,
    ) -> None:
        backend = matplotlib.get_backend()

        coordinates: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = self.get_swarm_positions()

        with open("./stored_field/field.pickle", "rb") as f:
            figure = pickle.load(f)
        ax = plt.gca()

        x, y = 100, 100

        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))

        ax.scatter(
            coordinates[:, 0] * self._field_quality_scale,
            coordinates[:, 1] * self._field_quality_scale,
            marker='o',
            color='b',
            ls='',
            s=40,
        )

        ax.set_xlim(0, 10 * self._field_quality_scale)
        ax.set_ylim(0, 10 * self._field_quality_scale)
        ax.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            ax.annotate(
                label,
                (
                    coordinates[index][0] * self._field_quality_scale,
                    coordinates[index][1] * self._field_quality_scale,
                ),
                fontsize=10,
            )

        for coordinate in coordinates:
            circle = matplotlib.patches.Circle(
                coordinate * self._field_quality_scale,
                self._connection_radius * self._field_size * self._field_quality_scale,
                color="g",
                fill=False,
                linestyle="--",
            )
            ax.add_patch(circle)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.5)
        plt.close(figure)
