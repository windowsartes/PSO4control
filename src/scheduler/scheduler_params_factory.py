from pydantic import BaseModel

from src.scheduler.scheduler_params import SCHEDULER_PARAMS_REGISTER


class SchedulerParamsFactory:
    @staticmethod
    def construct(config) -> BaseModel:  # type: ignore
        return SCHEDULER_PARAMS_REGISTER[config["type"].lower()](**config["params"])
