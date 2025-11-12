from src.solvers.grey_wolf_optimization.grey_wolf_optimization_params import GreyWolfOptimizationBaseParams
from src.solvers.grey_wolf_optimization.grey_wolf_optimization_params_factory import SolverParamsFactory
from src.solvers.grey_wolf_optimization.grey_wolf_optimization import SOLVER_REGISTER, GreyWolfOptimizationBase


class SolverFactory:
    _params_factory: SolverParamsFactory = SolverParamsFactory()

    def construct(  # type: ignore
        self,
        config,
        field_size: float,
        field_quality_scale: float,
    ) -> GreyWolfOptimizationBase:
        params: GreyWolfOptimizationBaseParams = self._params_factory.construct(config)

        specification = config["specification"].lower().replace("_", "")

        return SOLVER_REGISTER[specification](params, field_size, field_quality_scale)
