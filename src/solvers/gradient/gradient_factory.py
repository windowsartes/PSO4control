import typing as tp

from src.solvers.gradient.gradient_params import GradientParams
from src.solvers.gradient.gradient_params_factory import SolverParamsFactory
from src.solvers.gradient.gradient import SOLVER_REGISTER, GradientMethodBase


class SolverFactory:
    _params_factory: SolverParamsFactory = SolverParamsFactory()

    def construct(
        self,
        config,
        field_size: float,
        field_quality_scale: float,
    ) -> tp.Type[GradientMethodBase]:
        params: tp.Type[GradientParams] = self._params_factory.construct(config)

        return SOLVER_REGISTER[config["specification"].lower()](params, field_size, field_quality_scale)
