from src.solvers.swarm.swarm_params import SwarmCentralizedParams
from src.solvers.swarm.swarm_params_factory import SolverParamsFactory
from src.solvers.swarm.swarm import SOLVER_REGISTER, SwarmBase


class SolverFactory:
    _params_factory: SolverParamsFactory = SolverParamsFactory()

    def construct(  # type: ignore
        self,
        config,
        field_size: float,
        field_quality_scale: float,
    ) -> SwarmBase:
        params: SwarmCentralizedParams = self._params_factory.construct(config)

        return SOLVER_REGISTER[config["specification"].lower()](params, field_size, field_quality_scale)
