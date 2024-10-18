from pydantic import BaseModel

from src.scheduler.scheduler import SCHEDULER_REGISTER, SchedulerInteface
from src.scheduler.scheduler_params_factory import SchedulerParamsFactory


class SchedulerFactory:
    _params_factory: SchedulerParamsFactory = SchedulerParamsFactory()

    def construct(  # type: ignore
        self,
        config,
    ) -> SchedulerInteface:
        params: BaseModel = self._params_factory.construct(config)
        scheduler: SchedulerInteface = SCHEDULER_REGISTER[config["type"].lower()](params)

        return scheduler
