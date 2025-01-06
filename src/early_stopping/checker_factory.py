from pydantic import BaseModel

from src.early_stopping.stopping_params_factory import StoppingParamsFactory
from src.early_stopping.checker import EARLY_STOP_CHECKER_REGISTER, EarlyStopCheckerInterface


class EarlyStopCheckerFactory:
    _params_factory: StoppingParamsFactory = StoppingParamsFactory()

    def construct(  # type: ignore
        self,
        config,
    ) -> EarlyStopCheckerInterface:
        checker_params: BaseModel = self._params_factory.construct(config)

        return EARLY_STOP_CHECKER_REGISTER[config["type"].lower()](checker_params)
