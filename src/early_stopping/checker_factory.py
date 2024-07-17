import typing as tp

from pydantic import BaseModel

from src.early_stopping.stopping_params_factory import StoppingParamsFactory
from src.early_stopping.checker import EARLY_STOP_CHECKER_REGISTER, EarlyStopCheckerInterface


class EarlyStopCheckerFactory:
    def __init__(self):
        self._params_factory: StoppingParamsFactory = StoppingParamsFactory()

    def construct(self, params_config) -> tp.Type[EarlyStopCheckerInterface]:
        checker_params: tp.Type[BaseModel] = self._params_factory.construct(params_config)
        
        return EARLY_STOP_CHECKER_REGISTER[params_config["type"]](checker_params)
