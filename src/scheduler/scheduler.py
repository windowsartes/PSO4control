import typing as tp
from abc import ABC, abstractmethod

from src.scheduler import scheduler_params


class SchedulerInteface(ABC):
    @abstractmethod
    def step(self, *args, **kwargs):
        pass


SCHEDULER_REGISTER: dict[str, tp.Type[SchedulerInteface]] = {}


def scheduler(cls: tp.Type[SchedulerInteface]) -> tp.Type[SchedulerInteface]:
    SCHEDULER_REGISTER[cls.__name__[:-9].lower()] = cls
    return cls


@scheduler
class StepScheduler(SchedulerInteface):
    def __init__(self, hyperparameters: scheduler_params.StepSchedulerParams):
        super().__init__()

        self._hyperparameters: scheduler_params.StepSchedulerParams = hyperparameters
        self._current_step: int = 0

    def step(self, w: float):
        self._current_step += 1

        if self._current_step % self._hyperparameters.step_size == 0:
            self._current_step = 0
            return w * self._hyperparameters.gamma

        return w
