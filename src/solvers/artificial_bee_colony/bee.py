import typing as tp

import numpy as np

from src.solvers.grey_wolf_optimization.grey_wolf_optimization_params import SpawnParams


class EmployedBee:
    def __init__(
        self,
        field_size: float,
        spawn_params: SpawnParams,
        source_limit: int,
    ):
        self._field_size: float = field_size

        self._position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.empty((2))
        self._velocity: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.zeros((2))

        if spawn_params.type == "edge":
            spawn_edge: int = np.random.randint(4)
            if spawn_edge == 0:  # left
                self._position = np.array(
                    [
                        0.,
                        np.random.uniform(0, field_size),
                    ]
                )
            elif spawn_edge == 1:  # right
                self._position = np.array(
                    [
                        field_size,
                        np.random.uniform(0, field_size),
                    ]
                )
            elif spawn_edge == 2:  # top
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
        elif spawn_params.type == "spot":
            assert isinstance(spawn_params.factors.position, float)
            if spawn_params.spawn_edge == 0:  # left
                self._position = np.array(
                    [
                        0,
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position,
                        )
                    ]
                )
            elif spawn_params.spawn_edge == 1:  # right
                self._position = np.array(
                    [
                        field_size,
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position,
                        )
                    ]
                )
            elif spawn_params.spawn_edge == 2:  # top
                self._position = np.array(
                    [
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position
                        ),
                        0,
                    ]
                )
            elif spawn_params.spawn_edge == 3:  # bottom
                self._position = np.array(
                    [
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position
                        ),
                        field_size,
                    ]
                )
        elif spawn_params.type == "arc":
            assert isinstance(spawn_params.start_position, float)
            assert isinstance(spawn_params.finish_position, float)

            if spawn_params.start_edge == 0:  # left
                start_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.array(
                    [
                        0,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.start_edge == 1:  # right
                start_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.spawn_edge == 2:  # top
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        0,
                    ]
                )
            elif spawn_params.spawn_edge == 3:  # bottom
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("spwan edge must be int from 0 to 3")

            if spawn_params.finish_edge == 0:  # left
                finish_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.array(
                    [
                        0,
                        field_size * spawn_params.finish_position,
                    ]
                )

            elif spawn_params.finish_edge == 1:  # right
                finish_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.finish_position,
                    ]
                )
            elif spawn_params.finish_edge == 2:  # top
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        0,
                    ]
                )
            elif spawn_params.finish_edge == 3:  # bottom
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("finish edge must be int from 0 to 3")

            t = np.random.uniform(0 + 0.05, 1 - 0.05)

            self._position = start_position + t * (finish_position - start_position)
        elif spawn_params.type == "landing":
            assert isinstance(spawn_params.start_position, float)
            assert isinstance(spawn_params.finish_position, float)
            assert isinstance(spawn_params.factors.landing, float)

            if spawn_params.start_edge == 0:  # left
                start_position = np.array(
                    [
                        0,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.start_edge == 1:  # right
                start_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.spawn_edge == 2:  # top
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        0,
                    ]
                )
            elif spawn_params.spawn_edge == 3:  # bottom
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("spwan edge must be int from 0 to 3")

            if spawn_params.finish_edge == 0:  # left
                finish_position = np.array(
                    [
                        0,
                        field_size * spawn_params.finish_position,
                    ]
                )
            elif spawn_params.finish_edge == 1:  # right
                finish_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.finish_position,
                    ]
                )
            elif spawn_params.finish_edge == 2:  # top
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        0,
                    ]
                )
            elif spawn_params.finish_edge == 3:  # bottom
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("finish edge must be int from 0 to 3")

            self._position = start_position + spawn_params.landing_position * (finish_position - start_position)

            t_x: float = np.random.uniform(0, 1)
            t_y: float = np.random.uniform(0, 1)

            self._position += np.array(
                [
                    t_x * field_size / spawn_params.factors.landing,
                    t_y * field_size / spawn_params.factors.landing,
                ]
            )
        else:
            raise ValueError("spaen type must be 'spot', 'edge', 'arc' or 'landing'")

        self._score: float = 0.

        self._path_length: float = 0.

        self._velocity_factor: float = spawn_params.factors.velocity

        self._source_trials_limit = source_limit
        self._source_trials = 1

        self._best_score: float = self._score
        self._best_position = self._position.copy()

    def move(
        self,
        possible_next_position: np.ndarray,
        possible_next_score: float,
        field_size: float,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        if possible_next_score > self.score:
            self._velocity = possible_next_position - self._position

            if np.linalg.norm(self._velocity) > (field_size / self._velocity_factor):
                self._velocity = (self._velocity / np.linalg.norm(self._velocity)) * (field_size / self._velocity_factor)

            self._position = self._velocity + self._position

            self._source_trials = 1
        else:
            self._velocity = np.zeros_like(self._velocity)

            self._source_trials += 1

            if self._source_trials == self._source_trials_limit:
                angle = np.random.uniform(0, 2*np.pi)
                
                self._velocity = np.array(
                    [
                        (field_size / self._velocity_factor) * np.cos(angle),
                        (field_size / self._velocity_factor) * np.sin(angle),
                    ]
                )

                self._position = self._velocity + self._position

        self._path_length += float(np.linalg.norm(self._velocity))
        # if np.linalg.norm(self._velocity) > (field_size / self._velocity_factor):
        # self._path_length += np.min([np.linalg.norm(self._velocity), (field_size / self._velocity_factor)])

        return self._position

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, new_value: float) -> None:
        self._score = new_value

        if self._score > self._best_score:
            self._best_score = self._score
            self._best_position = self._position

    @property
    def best_score(self) -> float:
        return self._best_score
    
    @property
    def best_position(self) -> float:
        return self._best_position

    @property
    def position(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        return self._position

    @position.setter
    def position(
        self,
        new_value: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
    ) -> None:
        self._position = new_value

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def velocity(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        return self._velocity
    

class OnlookerBee:
    def __init__(
        self,
        field_size: float,
        spawn_params: SpawnParams,
    ):
        self._field_size: float = field_size

        self._position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.empty((2))
        self._velocity: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.zeros((2))

        if spawn_params.type == "edge":
            spawn_edge: int = np.random.randint(4)
            if spawn_edge == 0:  # left
                self._position = np.array(
                    [
                        0.,
                        np.random.uniform(0, field_size),
                    ]
                )
            elif spawn_edge == 1:  # right
                self._position = np.array(
                    [
                        field_size,
                        np.random.uniform(0, field_size),
                    ]
                )
            elif spawn_edge == 2:  # top
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
        elif spawn_params.type == "spot":
            assert isinstance(spawn_params.factors.position, float)
            if spawn_params.spawn_edge == 0:  # left
                self._position = np.array(
                    [
                        0,
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position,
                        )
                    ]
                )
            elif spawn_params.spawn_edge == 1:  # right
                self._position = np.array(
                    [
                        field_size,
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position,
                        )
                    ]
                )
            elif spawn_params.spawn_edge == 2:  # top
                self._position = np.array(
                    [
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position
                        ),
                        0,
                    ]
                )
            elif spawn_params.spawn_edge == 3:  # bottom
                self._position = np.array(
                    [
                        np.random.uniform(
                            field_size / 2 - field_size / spawn_params.factors.position,
                            field_size / 2 + field_size / spawn_params.factors.position
                        ),
                        field_size,
                    ]
                )
        elif spawn_params.type == "arc":
            assert isinstance(spawn_params.start_position, float)
            assert isinstance(spawn_params.finish_position, float)

            if spawn_params.start_edge == 0:  # left
                start_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.array(
                    [
                        0,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.start_edge == 1:  # right
                start_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.spawn_edge == 2:  # top
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        0,
                    ]
                )
            elif spawn_params.spawn_edge == 3:  # bottom
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("spwan edge must be int from 0 to 3")

            if spawn_params.finish_edge == 0:  # left
                finish_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]] = np.array(
                    [
                        0,
                        field_size * spawn_params.finish_position,
                    ]
                )

            elif spawn_params.finish_edge == 1:  # right
                finish_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.finish_position,
                    ]
                )
            elif spawn_params.finish_edge == 2:  # top
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        0,
                    ]
                )
            elif spawn_params.finish_edge == 3:  # bottom
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("finish edge must be int from 0 to 3")

            t = np.random.uniform(0 + 0.05, 1 - 0.05)

            self._position = start_position + t * (finish_position - start_position)
        elif spawn_params.type == "landing":
            assert isinstance(spawn_params.start_position, float)
            assert isinstance(spawn_params.finish_position, float)
            assert isinstance(spawn_params.factors.landing, float)

            if spawn_params.start_edge == 0:  # left
                start_position = np.array(
                    [
                        0,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.start_edge == 1:  # right
                start_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.start_position,
                    ]
                )
            elif spawn_params.spawn_edge == 2:  # top
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        0,
                    ]
                )
            elif spawn_params.spawn_edge == 3:  # bottom
                start_position = np.array(
                    [
                        field_size * spawn_params.start_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("spwan edge must be int from 0 to 3")

            if spawn_params.finish_edge == 0:  # left
                finish_position = np.array(
                    [
                        0,
                        field_size * spawn_params.finish_position,
                    ]
                )
            elif spawn_params.finish_edge == 1:  # right
                finish_position = np.array(
                    [
                        field_size,
                        field_size * spawn_params.finish_position,
                    ]
                )
            elif spawn_params.finish_edge == 2:  # top
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        0,
                    ]
                )
            elif spawn_params.finish_edge == 3:  # bottom
                finish_position = np.array(
                    [
                        field_size * spawn_params.finish_position,
                        field_size,
                    ]
                )
            else:
                raise ValueError("finish edge must be int from 0 to 3")

            self._position = start_position + spawn_params.landing_position * (finish_position - start_position)

            t_x: float = np.random.uniform(0, 1)
            t_y: float = np.random.uniform(0, 1)

            self._position += np.array(
                [
                    t_x * field_size / spawn_params.factors.landing,
                    t_y * field_size / spawn_params.factors.landing,
                ]
            )
        else:
            raise ValueError("spawn type must be 'spot', 'edge', 'arc' or 'landing'")

        self._score: float = 0.

        self._path_length: float = 0.

        self._velocity_factor: float = spawn_params.factors.velocity

        self._best_score = self._score
        self._best_position = self._position.copy()

    def move(
        self,
        possible_next_position: np.ndarray,
        possible_next_score: float,
        field_size: float,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        if possible_next_position is None and possible_next_score is None:
            angle = np.random.uniform(0, 2*np.pi)
                
            self._velocity = np.array(
                [
                    (field_size / self._velocity_factor) * np.cos(angle),
                    (field_size / self._velocity_factor) * np.sin(angle),
                ]
            )

            self._position = self._velocity + self._position
        elif possible_next_score > self._score:
            self._velocity = possible_next_position - self._position

            if np.linalg.norm(self._velocity) > (field_size / self._velocity_factor):
                self._velocity = (self._velocity / np.linalg.norm(self._velocity)) * (field_size / self._velocity_factor)

            self._position = self._velocity + self._position
        else:
            self._velocity = np.zeros_like(self._velocity)

        # self._path_length += np.min([np.linalg.norm(self._velocity), (field_size / self._velocity_factor)])
        self._path_length += float(np.linalg.norm(self._velocity))

        return self._position

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, new_value: float) -> None:
        self._score = new_value

        if self._score > self._best_score:
            self._best_score = self._score
            self._best_position = self._position

    @property
    def best_score(self) -> float:
        return self._best_score
    
    @property
    def best_position(self) -> float:
        return self._best_position

    @property
    def position(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        return self._position

    @position.setter
    def position(
        self,
        new_value: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
    ) -> None:
        self._position = new_value

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def velocity(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        return self._velocity
