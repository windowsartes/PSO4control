import typing as tp

from pydantic import BaseModel

from src.early_stopping.stopping_params import STOPPING_PARAMS_REGISTER


class StoppingParamsFactory:
    @staticmethod
    def construct(params_config) -> tp.Type[BaseModel]:
        return STOPPING_PARAMS_REGISTER[params_config["type"].lower()](**params_config["params"])
