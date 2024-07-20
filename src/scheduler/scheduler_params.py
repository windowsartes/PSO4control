import typing as tp

from pydantic import BaseModel


SCHEDULER_PARAMS_REGISTER: dict[str, tp.Type[BaseModel]] = {}

def scheduler_hyperparameters(cls: tp.Type[BaseModel]) -> tp.Type[BaseModel]:
    SCHEDULER_PARAMS_REGISTER[cls.__name__[:-15].lower()] = cls
    return cls


@scheduler_hyperparameters
class StepSchedulerParams(BaseModel):
    step_size: int
    gamma: float
