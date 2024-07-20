import typing as tp

from pydantic import BaseModel

from src.early_stopping.stopping_params_factory import StoppingParamsFactory
from src.early_stopping.checker import EARLY_STOP_CHECKER_REGISTER, EarlyStopCheckerInterface


class EarlyStopCheckerFactory:
    _params_factory: StoppingParamsFactory = StoppingParamsFactory()

    def construct(self, config) -> tp.Type[EarlyStopCheckerInterface]:
        checker_params: tp.Type[BaseModel] = self._params_factory.construct(config)
        
        return EARLY_STOP_CHECKER_REGISTER[config["type"].lower()](checker_params)
