import typing as tp

from src.scheduler.scheduler import (
    SCHEDULER_HYPERPARAMETERS_REGISTER,
    SCHEDULER_REGISTER,
    SchedulerInteface,
    SchedulerHyperparameters
)


class SchedulerFactory:
    def construct(self, scheduler_config) -> tp.Type[SchedulerInteface]:
        params: tp.Type[SchedulerHyperparameters] = \
            SCHEDULER_HYPERPARAMETERS_REGISTER[scheduler_config["type"]](**scheduler_config["params"])
        scheduler: tp.Type[SchedulerInteface] = SCHEDULER_REGISTER[scheduler_config["type"]](params)

        return scheduler
