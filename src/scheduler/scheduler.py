import typing as tp
from abc import ABC, abstractmethod

from pydantic import BaseModel


class SchedulerInteface(ABC):
    @abstractmethod
    def step(self, *args, **kwargs):
        pass

SCHEDULER_REGISTER: dict[str, tp.Type[SchedulerInteface]] = {}

def scheduler(cls: tp.Type[SchedulerInteface]) -> tp.Type[SchedulerInteface]:
    SCHEDULER_REGISTER[cls.__name__] = cls
    return cls


class SchedulerHyperparameters(BaseModel):
    pass

SCHEDULER_HYPERPARAMETERS_REGISTER: dict[str, tp.Type[SchedulerHyperparameters]] = {}

def scheduler_hyperparameters(cls: tp.Type[SchedulerHyperparameters]) -> tp.Type[SchedulerHyperparameters]:
    SCHEDULER_HYPERPARAMETERS_REGISTER[cls.__name__[:-15]] = cls
    return cls

@scheduler_hyperparameters
class StepSchedulerHyperparameters(SchedulerHyperparameters):
    step_size: int
    gamma: float

@scheduler
class StepScheduler(SchedulerInteface):
    def __init__(self, hyperparameters: StepSchedulerHyperparameters):
        super().__init__()

        self._hyperparameters: StepSchedulerHyperparameters = hyperparameters
        self._current_step: int = 0

        print(self._hyperparameters.step_size)

    def step(self, w: float):
        self._current_step+=1
        
        if self._current_step % self._hyperparameters.step_size == 0:
            self._current_step = 0
            return w * self._hyperparameters.gamma

        return w
