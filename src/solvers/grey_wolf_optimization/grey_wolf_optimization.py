import pickle
import random
import typing as tp
from abc import abstractmethod

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.answer.answer import Point
from src.solvers.solver_interface import SolverInterface
from src.solvers.grey_wolf_optimization.wolf import Wolf, WolfImproved
from src.solvers.grey_wolf_optimization import grey_wolf_optimization_params

# matplotlib.use('TKAgg')


class GreyWolfOptimizationBaseInterface:
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
    def get_wolves_position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def show(
        self,
        title: str,
    ) -> None:
        pass


SOLVER_REGISTER: dict[str, tp.Type["GreyWolfOptimizationBase"]] = {}


def solver(
    cls: tp.Type["GreyWolfOptimizationBase"],
) -> tp.Type["GreyWolfOptimizationBase"]:
    SOLVER_REGISTER[cls.__name__.lower()] = cls
    return cls


@solver
class GreyWolfOptimizationBase(GreyWolfOptimizationBaseInterface):
    def __init__(
        self,
        params: grey_wolf_optimization_params.GreyWolfOptimizationBaseParams,
        field_size: float,
        field_quality_scale: float,
    ) -> None:
        self._wolves: list[Wolf] = []
        for _ in range(params.n_wolves):
            self._wolves.append(
                Wolf(
                    field_size,
                    params.spawn,
                    params.a_coef,
                )
            )

        self._field_size: float = field_size
        self._field_quality_scale: float = field_quality_scale

        self._connection_radius: float = params.connection_radius
        self._connection_dropout_probability: float = params.connection_dropout_probability

        self._a_coef_original: float = params.a_coef

        self._alpha_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves
        self._beta_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves
        self._delta_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves


    def get_wolves_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        positions: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((len(self._wolves), 2), dtype=np.double)

        for index, wolf in enumerate(self._wolves):
            positions[index] = wolf.position

        return positions
    

    @property
    def particles(self) -> list[Wolf]:
        return self._wolves

    def correct_positions(
        self,
        field_size: float,
    ) -> None:
        for i in range(len(self._wolves)):
            self._wolves[i].position[0] = max(self._wolves[i].position[0], 0)
            self._wolves[i].position[0] = min(self._wolves[i].position[0], field_size)

            self._wolves[i].position[1] = max(self._wolves[i].position[1], 0)
            self._wolves[i].position[1] = min(self._wolves[i].position[1], field_size)
    
    def get_position_error(
        self,
        answer_point: Point,
        field_size: float,
    ) -> float:
        weights = np.array([wolf.score for wolf in self._wolves])
        weights /= np.sum(weights)

        if len(weights.shape) == 1:
            weights = weights[..., None]

        positions = np.array([wolf.position for wolf in self._wolves])
        consensus_position = np.sum(weights * positions, axis=0)

        assert consensus_position.shape == (2,)

        return np.linalg.norm(consensus_position - np.array((answer_point.x, answer_point.y))) / field_size

    def get_path_length(
        self,
    ) -> float:
        return sum([p.path_length for p in self._wolves])

    def update_scores(
        self,
        wolf_pack_scores: list[float],
    ) -> None:
        for i in range(len(self._wolves)):
            self._wolves[i].score = wolf_pack_scores[i]

        for i in range(len(self._wolves)):
            positions_and_scores = []

            for j in range(len(self._wolves)):
                if all([
                    random.uniform(0, 1) > self._connection_dropout_probability,
                    np.linalg.norm(self._wolves[i].position - self._wolves[j].position) <
                        self._connection_radius * self._field_size,
                ]):
                    positions_and_scores.append((self._wolves[j].position, j, self._wolves[j].score))
                    positions_and_scores = sorted(positions_and_scores, key = lambda x: x[-1], reverse=True)

                    if len(positions_and_scores) >= 3:
                        self._alpha_positions[i] = positions_and_scores[0][0]
                        self._beta_positions[i] = positions_and_scores[1][0]
                        self._delta_positions[i] = positions_and_scores[2][0]
                    elif len(positions_and_scores) == 2:
                        self._alpha_positions[i] = positions_and_scores[0][0]
                        self._beta_positions[i] = positions_and_scores[0][0]
                        self._delta_positions[i] = positions_and_scores[1][0]
                    else:
                        self._alpha_positions[i] = positions_and_scores[0][0]
                        self._beta_positions[i] = positions_and_scores[0][0]
                        self._delta_positions[i] = positions_and_scores[0][0]

    def turn(self) -> None:
        for i in range(len(self._wolves)):
            self._wolves[i].move(
                self._alpha_positions[i],
                self._beta_positions[i],
                self._delta_positions[i],
                self._field_size,
            )

    @property
    def wolves(self) -> list[Wolf]:
        return self._wolves

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


@solver
class GreyWolfOptimizationImproved(GreyWolfOptimizationBase):
    def __init__(
        self,
        params: grey_wolf_optimization_params.GreyWolfOptimizationBaseParams,
        field_size: float,
        field_quality_scale: float,
    ) -> None:
        self._wolves: list[WolfImproved] = []
        for _ in range(params.n_wolves):
            self._wolves.append(
                WolfImproved(
                    field_size,
                    params.spawn,
                    params.a_coef,
                )
            )

        self._field_size: float = field_size
        self._field_quality_scale: float = field_quality_scale

        self._connection_radius: float = params.connection_radius
        self._connection_dropout_probability: float = params.connection_dropout_probability

        self._a_coef_original: float = params.a_coef

        self._alpha_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves
        self._beta_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves
        self._delta_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves

        self._alpha_scores: list[float] = [0] * params.n_wolves
        self._beta_scores: list[float] = [0] * params.n_wolves
        self._delta_scores: list[float] = [0] * params.n_wolves

        self._r1_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves
        self._r2_positions: list[np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]] = \
            [np.empty((2))] * params.n_wolves

    def update_scores(
        self,
        wolf_pack_scores: list[float],
    ) -> None:
        for i in range(len(self._wolves)):
            self._wolves[i].score = wolf_pack_scores[i]

        for i in range(len(self._wolves)):
            positions_and_scores = []

            for j in range(len(self._wolves)):
                if all([
                    random.uniform(0, 1) > self._connection_dropout_probability,
                    np.linalg.norm(self._wolves[i].position - self._wolves[j].position) <
                        self._connection_radius * self._field_size,
                ]):
                    positions_and_scores.append((self._wolves[j].position, j, self._wolves[j].score))
                    positions_and_scores = sorted(positions_and_scores, key = lambda x: x[-1], reverse=True)

                    if len(positions_and_scores) >= 3:
                        self._alpha_positions[i] = positions_and_scores[0][0]
                        self._alpha_scores[i] = positions_and_scores[0][-1]

                        self._beta_positions[i] = positions_and_scores[1][0]
                        self._beta_scores[i] = positions_and_scores[1][-1]

                        self._delta_positions[i] = positions_and_scores[2][0]
                        self._delta_scores[i] = positions_and_scores[2][-1]

                        r1_index, r2_index = np.random.choice(range(len(positions_and_scores)), replace=False, size=2)

                        self._r1_positions[i] = positions_and_scores[r1_index][0]
                        self._r2_positions[i] = positions_and_scores[r2_index][0]
                    elif len(positions_and_scores) == 2:
                        self._alpha_positions[i] = positions_and_scores[0][0]
                        self._alpha_scores[i] = positions_and_scores[0][-1]

                        self._beta_positions[i] = positions_and_scores[0][0]
                        self._beta_scores[i] = positions_and_scores[0][-1]

                        self._delta_positions[i] = positions_and_scores[1][0]
                        self._delta_scores[i] = positions_and_scores[1][-1]

                        self._r1_positions[i] = positions_and_scores[0][0]
                        self._r2_positions[i] = positions_and_scores[1][0]
                    else:
                        self._alpha_positions[i] = positions_and_scores[0][0]
                        self._alpha_scores[i] = positions_and_scores[0][-1]

                        self._beta_positions[i] = positions_and_scores[0][0]
                        self._beta_scores[i] = positions_and_scores[0][-1]

                        self._delta_positions[i] = positions_and_scores[0][0]
                        self._delta_scores[i] = positions_and_scores[0][-1]

                        self._r1_positions[i] = positions_and_scores[0][0]
                        self._r2_positions[i] = positions_and_scores[0][0]

    def turn(self) -> None:
        for i in range(len(self._wolves)):
            self._wolves[i].move(
                self._alpha_positions[i],
                self._beta_positions[i],
                self._delta_positions[i],
                self._alpha_scores[i],
                self._beta_scores[i],
                self._delta_scores[i],
                self._r1_positions[i],
                self._r2_positions[i],
                self._field_size,
            )
