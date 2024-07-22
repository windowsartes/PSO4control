from pydantic import BaseModel


STOPPING_PARAMS_REGISTER: dict[str, BaseModel] = {}


def stopping_params(
    cls: BaseModel,
) -> BaseModel:
    STOPPING_PARAMS_REGISTER[cls.__name__[:-14].lower()] = cls
    return cls


class StoppingParams(BaseModel):
    epsilon: float
    ratio: float


@stopping_params
class SwarmStoppingParams(BaseModel):
    coordinate: StoppingParams
    velocity: StoppingParams


@stopping_params
class GradientLiftStoppingParams(BaseModel):
    velocity: StoppingParams
