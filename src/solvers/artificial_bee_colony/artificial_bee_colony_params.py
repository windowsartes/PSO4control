import typing as tp

from pydantic import BaseModel


SOLVER_PARAMS_REGISTER: dict[str, tp.Type["ArtificialBeeColonyBaseParams"]] = {}


def solver_params(
    cls: tp.Type["ArtificialBeeColonyBaseParams"],
) -> tp.Type["ArtificialBeeColonyBaseParams"]:
    SOLVER_PARAMS_REGISTER[cls.__name__[:-6].lower()] = cls
    return cls


class Factors(BaseModel):
    velocity: float
    position: tp.Optional[float] = None
    landing: tp.Optional[float] = None


class SpawnParams(BaseModel):
    type: str
    spawn_edge: tp.Optional[int] = None
    start_edge: tp.Optional[int] = None
    finish_edge: tp.Optional[int] = None
    start_position: tp.Optional[float] = None
    finish_position: tp.Optional[float] = None
    landing_position: tp.Optional[float] = None
    factors: Factors


@solver_params
class ArtificialBeeColonyBaseParams(BaseModel):
    n_iterations: int
    n_bees: int
    spawn: SpawnParams
    source_limit: float
    connection_radius: float
    connection_dropout_probability: float


@solver_params
class ArtificialBeeColonyAdaptiveExplorationParams(ArtificialBeeColonyBaseParams):
    pass