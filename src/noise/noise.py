import typing as tp
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from src.answer.answer import Answer


class NoiseHyperparameters(BaseModel):
    loc: float
    scale: float


class NoiseInterface(ABC):
    @abstractmethod
    def get_noise(  # type: ignore
        self,
        *args,
        **kwargs,
    ) -> float:
        pass


class NoiseBase(NoiseInterface):
    def __init__(
        self,
        answer: Answer,
        params: NoiseHyperparameters,
    ):
        super().__init__()

        self._answer: Answer = answer
        self._params: NoiseHyperparameters = params


NOISE_REGISTER: dict[str, tp.Type[NoiseBase]] = {}


def noise(cls: tp.Type[NoiseBase]) -> tp.Type[NoiseBase]:
    NOISE_REGISTER[cls.__name__[:-5].lower()] = cls
    return cls


@noise
class InverseDistanceNoise(NoiseBase):
    def __init__(
        self,
        answer: Answer,
        params: NoiseHyperparameters,
    ):
        super().__init__(
            answer,
            params,
        )

    def _get_closest_answer(
        self,
        particle_position: np.ndarray[tp.Any, np.dtype[np.float64]],
    ) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        closest_answer: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            np.array([self._answer.answers[0].x, self._answer.answers[0].y])
        closest_distance: float = float(np.linalg.norm(particle_position - closest_answer))

        for i in range(1, len(self._answer.answers)):
            current_answer: np.ndarray[tp.Any, np.dtype[np.float64]] = \
                np.array([self._answer.answers[i].x, self._answer.answers[i].y])
            current_distance: float = float(np.linalg.norm(particle_position - current_answer))

            if current_distance < closest_distance:
                closest_distance = current_distance
                closest_answer = current_answer

        return closest_answer

    def get_noise(
        self,
        particle_position: np.ndarray[tp.Any, np.dtype[np.float64]],
    ) -> float:
        closest_answer: np.ndarray[tp.Any, np.dtype[np.float64]] = \
            self._get_closest_answer(particle_position)

        return np.random.normal(
            0,
            np.linalg.norm(particle_position - closest_answer),
            size=1,
        )[0] * self._params.scale + self._params.loc


@noise
class RelativeVarianceNoise(NoiseBase):
    def __init__(
        self,
        answer: Answer,
        params: NoiseHyperparameters,
    ):
        super().__init__(
            answer,
            params,
        )

    def get_noise(
        self,
        field_value: float,
    ) -> float:
        return float(
            np.random.normal(
                self._params.loc,
                abs(field_value) * self._params.scale,
                size=1,
            )[0]
        )
