import importlib
import typing as tp

from src.solvers.solver_interface import SolverInterface


class SolverFactory:
    def construct(
        self,
        config,
        filed_size: float,
        field_quality_scale: float,
    ) -> tp.Type[SolverInterface]:
        solver_factory_module = importlib.import_module(f"src.solvers.{config['type']}.{config['type']}_factory")

        return solver_factory_module.SolverFactory().construct(config, filed_size, field_quality_scale)
