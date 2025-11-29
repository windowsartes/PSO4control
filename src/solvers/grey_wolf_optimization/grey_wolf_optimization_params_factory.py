import random

from src.solvers.grey_wolf_optimization.grey_wolf_optimization_params import SOLVER_PARAMS_REGISTER, GreyWolfOptimizationBaseParams


class SolverParamsFactory:
    def construct(self, config) -> GreyWolfOptimizationBaseParams:  # type: ignore
        specification = config["specification"].lower().replace("_", "")
        params: GreyWolfOptimizationBaseParams = SOLVER_PARAMS_REGISTER[specification](**config["params"])

        if params.spawn.type == "spot":
            params.spawn.spawn_edge = random.randint(0, 3)

        if params.spawn.type == "arc" or params.spawn.type == "landing":
            params.spawn.start_edge = random.randint(0, 1)
            all_edges: set[int] = set(range(0, 4))
            all_edges.remove(params.spawn.start_edge)
            params.spawn.finish_edge = random.choice(list(all_edges))
            params.spawn.start_position = random.uniform(0 + 0.05, 1 - 0.05)
            params.spawn.finish_position = random.uniform(0 + 0.05, 1 - 0.05)

        if params.spawn.type == "landing":
            params.spawn.landing_position = random.uniform(0 + 0.05, 1 - 0.05)

        return params
