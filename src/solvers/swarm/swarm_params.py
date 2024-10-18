import typing as tp

from pydantic import BaseModel


SOLVER_PARAMS_REGISTER: dict[str, tp.Type["SwarmCentralizedParams"]] = {}


def solver_params(
    cls: tp.Type["SwarmCentralizedParams"],
) -> tp.Type["SwarmCentralizedParams"]:
    SOLVER_PARAMS_REGISTER[cls.__name__[5:-6].lower()] = cls
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


class ParticleCoefficients(BaseModel):
    w: float
    c1: float
    c2: float


@solver_params
class SwarmCentralizedParams(BaseModel):
    n_iterations: int
    n_particles: int
    spawn: SpawnParams
    coefficients: ParticleCoefficients


@solver_params
class SwarmDecentralizedParams(SwarmCentralizedParams):
    connection_radius: float
