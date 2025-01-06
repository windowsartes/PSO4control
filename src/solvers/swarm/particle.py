import typing as tp

import numpy as np

from src.solvers.swarm.swarm_params import ParticleCoefficients, SpawnParams


class Particle:
    """
    Частица умеет только двигаться, после каждого движения она возвращает своё новое положение.
    Сцена его корректирует и прописывает новое положение частицы с помощью её сеттера.
    Сцена же считает скор, полученный частицей и прсотавляет его. Лучше положение частицы и её лучший скор
    также обновляет сцена. Лучший скор и лучшее положение роя также обновляетс сцена.
    """
    def __init__(
        self,
        field_size: float,
        spawn_params: SpawnParams,
        coefficients: ParticleCoefficients,
    ):
        self._field_size: float = field_size

        self._position: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((2))
        self._velocity: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((2))

        if spawn_params.type == "full_location":
            self._position = np.array(
                [
                    np.random.uniform(0, field_size),
                    np.random.uniform(0, field_size),
                ]
            )
            self._velocity = np.array(
                [
                    np.random.uniform(-field_size, field_size),
                    np.random.uniform(-field_size, field_size),
                ]
            )
        elif spawn_params.type == "edge":
            spawn_edge: int = np.random.randint(4)
            if spawn_edge == 0:  # left
                self._position = np.array(
                    [
                        0.,
                        np.random.uniform(0, field_size),
                    ]
                )
                self._velocity = np.array(
                    [
                        np.random.uniform(0, field_size),
                        np.random.uniform(-field_size, field_size),
                    ]
                )
            elif spawn_edge == 1:  # right
                self._position = np.array(
                    [
                        field_size,
                        np.random.uniform(0, field_size),
                    ]
                )
                self._velocity = np.array(
                    [
                        np.random.uniform(-field_size, 0),
                        np.random.uniform(-field_size, field_size),
                    ]
                )
            elif spawn_edge == 2:  # top
                self._position = np.array(
                    [
                        np.random.uniform(0, field_size),
                        0,
                    ]
                )
                self._velocity = np.array(
                    [
                        np.random.uniform(-field_size, field_size),
                        np.random.uniform(0, field_size),
                    ]
                )
            else:  # bottom
                self._position = np.array(
                    [
                        np.random.uniform(0, field_size),
                        field_size,
                    ]
                )
                self._velocity = np.array(
                    [
                        np.random.uniform(-field_size, field_size),
                        np.random.uniform(-field_size, 0),
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
                self._velocity = np.array(
                    [
                        np.random.uniform(0, field_size),
                        np.random.uniform(-field_size, field_size),
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
                self._velocity = np.array(
                    [
                        np.random.uniform(-field_size, 0),
                        np.random.uniform(-field_size, field_size),
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
                self._velocity = np.array(
                    [
                        np.random.uniform(-field_size, field_size),
                        np.random.uniform(0, field_size),
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
                self._velocity = np.array(
                    [
                        np.random.uniform(-field_size, field_size),
                        np.random.uniform(-field_size, 0),
                    ]
                )
        elif spawn_params.type == "arc":
            assert isinstance(spawn_params.start_position, float)
            assert isinstance(spawn_params.finish_position, float)
            if spawn_params.start_edge == 0:  # left
                start_position: np.ndarray[tp.Any, np.dtype[np.float64]] = np.array(
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
                finish_position: np.ndarray[tp.Any, np.dtype[np.float64]] = np.array(
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
            self._velocity = np.array(
                [
                    np.random.uniform(-field_size, field_size),
                    np.random.uniform(-field_size, field_size),
                ]
            )
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
            self._velocity = np.array(
                    [
                        np.random.uniform(-field_size, field_size),
                        np.random.uniform(-field_size, field_size),
                    ]
                )

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

        self._best_score: float = 0.
        self._best_position: np.ndarray[tp.Any, np.dtype[np.float64]] = self._position

        self._path_length: float = 0.

        self._w: float = coefficients.w
        self._c1: float = coefficients.c1
        self._c2: float = coefficients.c2

        self._velocity_factor: float = spawn_params.factors.velocity

    def move(
        self,
        best_global_position: np.ndarray[tp.Any, np.dtype[np.float64]],
        field_size: float,
    ) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        r_personal: float = np.random.uniform()
        r_global: float = np.random.uniform()

        self._velocity = self._w * self._velocity + \
            self._c1 * r_personal * (self._best_position - self._position) + \
            self._c2 * r_global * (best_global_position - self._position)

        if np.linalg.norm(self._velocity) > (field_size / self._velocity_factor):
            self._velocity = (self._velocity / np.linalg.norm(self._velocity)) * (field_size / self._velocity_factor)

        self._position = self._velocity + self._position
        self._path_length += float(np.linalg.norm(self._velocity))

        return self._position

    @property
    def best_score(self) -> float:
        return self._best_score

    @best_score.setter
    def best_score(self, new_value: float) -> None:
        self._best_score = new_value

    @property
    def best_position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._best_position

    @best_position.setter
    def best_position(
        self,
        new_value: np.ndarray[tp.Any, np.dtype[np.float64]],
    ) -> None:
        self._best_position = new_value

    @property
    def position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._position

    @position.setter
    def position(
        self,
        new_value: np.ndarray[tp.Any, np.dtype[np.float64]],
    ) -> None:
        self._position = new_value

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def velocity(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._velocity

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(
        self,
        new_value: float,
    ) -> None:
        self._w = new_value
