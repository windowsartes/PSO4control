import typing as tp

from pydantic import BaseModel

from src.scheduler.scheduler import SCHEDULER_REGISTER, SchedulerInteface
from src.scheduler.scheduler_params_factory import SchedulerParamsFactory


class SchedulerFactory:
    _params_factory: SchedulerParamsFactory = SchedulerParamsFactory()

    def construct(self, config) -> tp.Type[SchedulerInteface]:
        params: tp.Type[BaseModel] = self._params_factory.construct(config)
        scheduler: tp.Type[SchedulerInteface] = SCHEDULER_REGISTER[config["type"].lower()](params)

        return scheduler
