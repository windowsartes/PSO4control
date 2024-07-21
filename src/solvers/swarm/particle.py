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
        elif spawn_params.type == "edges":
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
        elif spawn_params.type == "small_area":
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
    def position(
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
