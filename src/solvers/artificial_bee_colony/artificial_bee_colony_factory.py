from src.solvers.artificial_bee_colony.artificial_bee_colony_params import ArtificialBeeColonyBaseParams
from src.solvers.artificial_bee_colony.artificial_bee_colony_params_factory import SolverParamsFactory
from src.solvers.artificial_bee_colony.artificial_bee_colony import SOLVER_REGISTER, ArtificialBeeColonyBase


class SolverFactory:
    _params_factory: SolverParamsFactory = SolverParamsFactory()

    def construct(  # type: ignore
        self,
        config,
        field_size: float,
        field_quality_scale: float,
    ) -> ArtificialBeeColonyBase:
        params: ArtificialBeeColonyBaseParams = self._params_factory.construct(config)

        specification = config["specification"].lower().replace("_", "")

        return SOLVER_REGISTER[specification](params, field_size, field_quality_scale)
