import typing as tp
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from src.early_stopping import stopping_params
from src.solvers.swarm.particle import Particle


class EarlyStopCheckerInterface(ABC):
    @abstractmethod
    def check(self, *args, **kwargs):
        pass


EARLY_STOP_CHECKER_REGISTER: dict[str, BaseModel] = {}


def checker(
    cls: tp.Type[EarlyStopCheckerInterface],
) -> tp.Type[EarlyStopCheckerInterface]:
    EARLY_STOP_CHECKER_REGISTER[cls.__name__[:-16].lower()] = cls
    return cls


@checker
class SwarmEarlyStopChecker(EarlyStopCheckerInterface):
    def __init__(
        self,
        params: stopping_params.SwarmStoppingParams,
    ):
        super().__init__()

        self._params: stopping_params.SwarmStoppingParams = params

    def check(
        self,
        particles: list[Particle],
    ) -> bool:
        return self.check_position(particles) or self.check_velocity(particles)

    def check_position(
        self,
        particles: list[Particle],
    ) -> bool:
        for i in range(len(particles)):
            close_points_count: int = 0
            for j in range(len(particles)):
                if np.linalg.norm(particles[i].position - particles[j].position) < self._params.coordinate.epsilon:
                    close_points_count += 1

                    if close_points_count > self._params.coordinate.ratio * len(particles):
                        return True

        return False

    def check_velocity(
        self,
        particles: list[Particle],
    ) -> bool:
        small_velocity_points_count: int = 0

        for i in range(len(particles)):
            if np.linalg.norm(particles[i].velocity) < self._params.velocity.epsilon:
                small_velocity_points_count += 1

            if small_velocity_points_count > self._params.velocity.ratio * len(particles):
                return True

        return False
