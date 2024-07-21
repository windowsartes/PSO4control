import typing as tp

from src.solvers.swarm.swarm_params import SOLVER_PARAMS_REGISTER, SwarmCentralizedParams


class SolverParamsFactory:
    def construct(self, config) -> tp.Type[SwarmCentralizedParams]:
        return SOLVER_PARAMS_REGISTER[config["specification"]](**config["params"])
