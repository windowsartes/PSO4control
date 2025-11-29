import random
import typing as tp
from abc import abstractmethod

import numpy as np

from src.answer.answer import Point
# from src.solvers.solver_interface import SolverInterface
from src.solvers.artificial_bee_colony.bee import EmployedBee, OnlookerBee
from src.solvers.artificial_bee_colony import artificial_bee_colony_params


'''
class ArtificialBeeColonyInterface:
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
    def get_bees_position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def get_bees_scores(self) -> list[float]:
        pass
'''

SOLVER_REGISTER: dict[str, tp.Type["ArtificialBeeColonyBase"]] = {}


def solver(
    cls: tp.Type["ArtificialBeeColonyBase"],
) -> tp.Type["ArtificialBeeColonyBase"]:
    SOLVER_REGISTER[cls.__name__.lower()] = cls
    return cls


@solver
class ArtificialBeeColonyBase:
    def __init__(
        self,
        params: artificial_bee_colony_params.ArtificialBeeColonyBaseParams,
        field_size: float,
        field_quality_scale: float,
    ) -> None:
        n_onlooker_bees = params.n_bees // 2
        n_employed_bees = params.n_bees - params.n_bees // 2

        self._employed_bees = []
        for _ in range(n_employed_bees):
            self._employed_bees.append(
                EmployedBee(
                    field_size,
                    params.spawn,
                    params.source_limit,
                )
            )

        self._onlooker_bees = []
        for _ in range(n_onlooker_bees):
            self._onlooker_bees.append(
                OnlookerBee(
                    field_size,
                    params.spawn,
                )
            )

        self._employed_bees_possible_positions = [0] * n_employed_bees
        self._employed_bees_possible_scores = [0] * n_employed_bees

        self._onlooker_bees_possible_positions = [0] * n_onlooker_bees
        self._onlooker_bees_possible_scores = [0] * n_onlooker_bees

        self._field_size: float = field_size
        self._field_quality_scale: float = field_quality_scale

        self._connection_radius: float = params.connection_radius
        self._connection_dropout_probability: float = params.connection_dropout_probability


    def get_bees_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        n_bees = len(self._employed_bees) + len(self._onlooker_bees)

        positions: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((n_bees, 2), dtype=np.double)

        for index, bee in enumerate(self._employed_bees + self._onlooker_bees):
            positions[index] = bee.position

        return positions[:len(self._employed_bees), ...], positions[len(self._employed_bees):, ...]
    

    def get_bees_scores(self) -> list[float]:
        scores = []

        for  bee in self._employed_bees + self._onlooker_bees:
            scores.append(bee.score)

        return scores

    def correct_positions(
        self,
        field_size: float,
    ) -> None:
        for i in range(len(self._employed_bees)):
            for j in range(self._employed_bees[i].position.shape[0]):
                self._employed_bees[i].position[j] = np.clip(self._employed_bees[i].position[j], a_min=0, a_max=field_size)

        for i in range(len(self._onlooker_bees)):
            for j in range(self._onlooker_bees[i].position.shape[0]):
                self._onlooker_bees[i].position[j] = np.clip(self._onlooker_bees[i].position[j], a_min=0, a_max=field_size)

    def get_position_error(
        self,
        answer_point: Point,
        field_size: float,
    ) -> float:
        weights = np.array([bee.best_score for bee in self._employed_bees + self._onlooker_bees])
        weights /= np.sum(weights)

        if len(weights.shape) == 1:
            weights = weights[..., None]

        positions = np.array([bee.best_position for bee in self._employed_bees + self._onlooker_bees])
        consensus_position = np.sum(weights * positions, axis=0)

        assert consensus_position.shape == (2,)

        return np.linalg.norm(consensus_position - np.array((answer_point.x, answer_point.y))) / field_size

    def get_path_length(
        self,
    ) -> float:
        return sum([p.path_length for p in self._employed_bees + self._onlooker_bees])

    def updated_employed_bees_scores_only(
        self,
        employed_bees_scores,
    ):
        for i in range(len(self._employed_bees)):
            self._employed_bees[i].score = employed_bees_scores[i]

    def update_employed_bees_scores(
        self,
        employed_bees_scores: list[float],
    ) -> None:
        for i in range(len(self._employed_bees)):
            self._employed_bees[i].score = employed_bees_scores[i]

        for i in range(len(self._employed_bees)):
            positions = []

            for j in range(len(self._employed_bees)):
                if all([
                    random.uniform(-1, 1) > self._connection_dropout_probability,
                    np.linalg.norm(self._employed_bees[i].position - self._employed_bees[j].position) <
                        self._connection_radius * self._field_size,
                    i != j,
                ]):
                    positions.append(self._employed_bees[j].position)

            if len(positions) == 0:
                chosen_positions = self._employed_bees[i].position
                chosen_score = employed_bees_scores[i]
            else:
                chosen_index = random.sample(range(len(positions)), 1)[0]
                chosen_positions = positions[chosen_index]
                chosen_score = employed_bees_scores[chosen_index]

            possible_next_position = self._employed_bees[i].position.copy()

            dim_index = random.sample(range(0, chosen_positions.size), 1)[0]

            shift = random.uniform(-1, 1) * (possible_next_position[dim_index] - chosen_positions[dim_index])
            possible_next_position[dim_index] += shift

            self._employed_bees_possible_positions[i] = possible_next_position
            self._employed_bees_possible_scores[i] = chosen_score

    def update_onlooker_bees_scores(
        self,
        onlooker_bees_scores: list[float],
    ) -> None:
        for i in range(len(self._onlooker_bees)):
            self._onlooker_bees[i].score = onlooker_bees_scores[i]

        for i in range(len(self._onlooker_bees)):
            positions_and_scores = []

            for j in range(len(self._employed_bees)):
                if all([
                    random.uniform(0, 1) > self._connection_dropout_probability,
                    np.linalg.norm(self._onlooker_bees[i].position - self._employed_bees[j].position) <
                        self._connection_radius * self._field_size,
                    True,
                ]):
                    positions_and_scores.append((self._employed_bees[j].position, self._employed_bees[j].score))

            if len(positions_and_scores) == 0:
                # self._onlooker_bees_possible_positions[i] = self._onlooker_bees[i].position
                # self._onlooker_bees_possible_scores[i] = onlooker_bees_scores[i]
                self._onlooker_bees_possible_positions[i] = None
                self._onlooker_bees_possible_scores[i] = None
            else:
                if len(positions_and_scores) == 1:
                    chosen_position, chosen_score = positions_and_scores[0]
                else:
                    positions = [position for position, _ in positions_and_scores]
                    scores = [score for _, score in positions_and_scores]

                    chosen_index = random.choices(range(len(positions)), weights=scores, k=1)[0]
                    chosen_position = positions[chosen_index]
                    chosen_score = scores[chosen_index]

                shift = self._onlooker_bees[i].position.copy()

                unique_values = list(range(chosen_position.size))
                dim_index = random.sample(list(unique_values), 1)[0]

                while shift[dim_index] == chosen_position[dim_index]:
                    unique_values.remove(dim_index)

                    if len(unique_values) == 0:
                        self._onlooker_bees_possible_positions[i] = None
                        self._onlooker_bees_possible_scores[i] = None

                        return

                    dim_index = random.sample(list(unique_values), 1)[0]

                shift[dim_index] += random.uniform(-1, 1) * (shift[dim_index] - chosen_position[dim_index])

                self._onlooker_bees_possible_positions[i] = self._onlooker_bees[i].position + shift
                self._onlooker_bees_possible_scores[i] = chosen_score

    def turn(self) -> None:
        for i in range(len(self._employed_bees)):
            self._employed_bees[i].move(
                self._employed_bees_possible_positions[i],
                self._employed_bees_possible_scores[i],
                self._field_size,
            )

        for i in range(len(self._onlooker_bees)):
            self._onlooker_bees[i].move(
                self._onlooker_bees_possible_positions[i],
                self._onlooker_bees_possible_scores[i],
                self._field_size,
            )

    @property
    def bees(self) -> list[EmployedBee | OnlookerBee]:
        return self._employed_bees + self._onlooker_bees


@solver
class ArtificialBeeColonyAdaptiveExploration(ArtificialBeeColonyBase):
    def update_employed_bees_scores(
        self,
        employed_bees_scores: list[float],
    ) -> None:
        for i in range(len(self._employed_bees)):
            self._employed_bees[i].score = employed_bees_scores[i]

        for i in range(len(self._employed_bees)):
            positions = []

            for j in range(len(self._employed_bees)):
                if all([
                    random.uniform(-1, 1) > self._connection_dropout_probability,
                    np.linalg.norm(self._employed_bees[i].position - self._employed_bees[j].position) <
                        self._connection_radius * self._field_size,
                    i != j,
                ]):
                    positions.append(self._employed_bees[j].position)

            if len(positions) == 0:
                chosen_positions = self._employed_bees[i].position
                chosen_score = employed_bees_scores[i]
            else:
                chosen_index = random.sample(range(len(positions)), 1)[0]
                chosen_positions = positions[chosen_index]
                chosen_score = employed_bees_scores[chosen_index]

            possible_next_position = self._employed_bees[i].position.copy()

            dim_index = random.sample(range(0, chosen_positions.size), 1)[0]

            shift = random.uniform(-1, 1) * (possible_next_position[dim_index] - chosen_positions[dim_index])
            possible_next_position[dim_index] += shift

            d = np.abs(possible_next_position[dim_index] - chosen_positions[dim_index])
            p = 1 if d == 0 else np.exp(-1/d)

            if r := np.random.random() > p:
                possible_next_position *= r

            self._employed_bees_possible_positions[i] = possible_next_position
            self._employed_bees_possible_scores[i] = chosen_score
