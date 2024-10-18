from pydantic import BaseModel

from src.early_stopping.stopping_params import STOPPING_PARAMS_REGISTER


class StoppingParamsFactory:
    @staticmethod
    def construct(params_config) -> BaseModel:  # type: ignore
        return STOPPING_PARAMS_REGISTER[params_config["type"].lower()](**params_config["params"])
