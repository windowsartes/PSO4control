import typing as tp

import numpy as np

from src.solvers.grey_wolf_optimization.grey_wolf_optimization_params import SpawnParams


class Wolf:
    def __init__(
        self,
        field_size: float,
        spawn_params: SpawnParams,
        a: float,
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

        self._a = a

    def move(
        self,
        alpha_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        beta_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        delta_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        field_size: float,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        r1, r2 = np.random.random(2)
        A1 = 2 * self._a * r1 - self._a
        C1 = 2 * r2

        r1, r2 = np.random.random(2)
        A2 = 2 * self._a * r1 - self._a
        C2 = 2 * r2

        r1, r2 = np.random.random(2)
        A3 = 2 * self._a * r1 - self._a
        C3 = 2 * r2

        D_alpha = abs(C1 * alpha_position - self._position)
        D_beta = abs(C2 * beta_position - self._position)
        D_delta = abs(C3 * delta_position - self._position)

        X1 = alpha_position - A1 * D_alpha
        X2 = beta_position - A2 * D_beta
        X3 = delta_position - A3 * D_delta

        predicted_next_position = (X1 + X2 + X3) / 3

        self._velocity = predicted_next_position - self._position

        if np.linalg.norm(self._velocity) > (field_size / self._velocity_factor):
            self._velocity = (self._velocity / np.linalg.norm(self._velocity)) * (field_size / self._velocity_factor)

        self._position = self._velocity + self._position
        self._path_length += float(np.linalg.norm(self._velocity))

        return self._position

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, new_value: float) -> None:
        self._score = new_value

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

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(
        self,
        new_value: float,
    ) -> None:
        self._a = new_value


class WolfImproved(Wolf):
    def move(
        self,
        alpha_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        beta_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        delta_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        alpha_score: float,
        beta_score: float,
        delta_score: float,
        r1_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        r2_position: np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]],
        field_size: float,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[tp.Any]]]:
        alpha_position = alpha_position + 2 * self._a * np.random.random() * (r1_position - r2_position)
        beta_position = beta_position + 2 * self._a * np.random.random() * (alpha_position - beta_position)
        delta_position = beta_position + 2 * self._a * np.random.random() * (alpha_position - delta_position)

        r1, r2 = np.random.random(2)
        A1 = 2 * self._a * r1 - self._a
        C1 = 2 * r2

        r1, r2 = np.random.random(2)
        A2 = 2 * self._a * r1 - self._a
        C2 = 2 * r2

        r1, r2 = np.random.random(2)
        A3 = 2 * self._a * r1 - self._a
        C3 = 2 * r2

        D_alpha = abs(C1 * alpha_position - self._position)
        D_beta = abs(C2 * beta_position - self._position)
        D_delta = abs(C3 * delta_position - self._position)

        X1 = alpha_position - A1 * D_alpha
        X2 = beta_position - A2 * D_beta
        X3 = delta_position - A3 * D_delta

        scores_sum = alpha_score + beta_score + delta_score

        w_alpha = alpha_score / scores_sum
        w_beta = beta_score / scores_sum
        w_delta = delta_score / scores_sum

        predicted_next_position = (X1 * w_alpha + X2 * w_beta + X3 * w_delta) / (w_alpha + w_beta + w_delta)

        self._velocity = predicted_next_position - self._position

        if np.linalg.norm(self._velocity) > (field_size / self._velocity_factor):
            self._velocity = (self._velocity / np.linalg.norm(self._velocity)) * (field_size / self._velocity_factor)

        self._position = self._velocity + self._position
        self._path_length += float(np.linalg.norm(self._velocity))

        return self._position