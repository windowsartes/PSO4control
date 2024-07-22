from pydantic import BaseModel


SOLVER_PARAMS_REGISTER: dict[str, BaseModel] = {}


def solver_params(
    cls: BaseModel,
) -> BaseModel:
    SOLVER_PARAMS_REGISTER[cls.__name__[:-6].lower()] = cls
    return cls


class GradientParams(BaseModel):
    n_iterations: int
    velocity_factor: float


@solver_params
class GradientLiftParams(GradientParams):
    pass


@solver_params
class NewtonsMethodParams(GradientParams):
    pass
