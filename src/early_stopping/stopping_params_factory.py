import typing as tp

from pydantic import BaseModel

from src.early_stopping.stopping_params import STOPPING_PARAMS_REGISTER


class StoppingParamsFactory:
    def construct(self, params_config) -> tp.Type[BaseModel]:
        return STOPPING_PARAMS_REGISTER[params_config["type"]](**params_config["params"])
