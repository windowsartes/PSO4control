import typing as tp

from src.solvers.gradient.gradient_params import SOLVER_PARAMS_REGISTER, GradientParams


class SolverParamsFactory:
    def construct(self, config) -> tp.Type[GradientParams]:
        return SOLVER_PARAMS_REGISTER[config["specification"].lower()](**config["params"])
