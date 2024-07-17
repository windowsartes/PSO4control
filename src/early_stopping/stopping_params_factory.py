import typing as tp

from pydantic import BaseModel

from src.early_stopping.stopping_params import STOPPING_PARAMS_REGISTER


class StoppingParamsFactory:
    def construct(self, params_config) -> tp.Type[BaseModel]:
        print(params_config["type"])
        return STOPPING_PARAMS_REGISTER[params_config["type"]](**params_config["params"])


f = StoppingParamsFactory()

config = {
    'type': 'Swarm',
    'params': {
        'coordinate': {
            'epsilon': 0.0001,
            'ratio': 0.75,
        },
        'velocity': {
            'epsilon': 0.0001,
            'ratio': 0.75,
        }
    }
}

p = f.construct(config)

print(p)