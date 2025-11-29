import typing as tp

from pydantic import BaseModel


SOLVER_PARAMS_REGISTER: dict[str, tp.Type["GreyWolfOptimizationBaseParams"]] = {}


def solver_params(
    cls: tp.Type["GreyWolfOptimizationBaseParams"],
) -> tp.Type["GreyWolfOptimizationBaseParams"]:
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
class GreyWolfOptimizationBaseParams(BaseModel):
    n_iterations: int
    n_wolves: int
    spawn: SpawnParams
    a_coef: float
    connection_radius: float
    connection_dropout_probability: float


@solver_params
class GreyWolfOptimizationImprovedParams(GreyWolfOptimizationBaseParams):
    pass
