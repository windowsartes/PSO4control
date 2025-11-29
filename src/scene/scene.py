import json
import typing as tp

import numpy as np

from src.answer.answer import Answer
from src.early_stopping.checker import EarlyStopCheckerInterface
from src.early_stopping.checker_factory import EarlyStopCheckerFactory
from src.field.field import FieldInterface
from src.field.field_factory import FieldFactory
from src.noise import noise
from src.noise.noise_factory import NoiseFactory
from src.scheduler.scheduler import SchedulerInteface
from src.scheduler.scheduler_factory import SchedulerFactory
from src.solvers.artificial_bee_colony import artificial_bee_colony
from src.solvers.grey_wolf_optimization import grey_wolf_optimization
from src.solvers.swarm import swarm
from src.solvers.solver_interface import SolverInterface
from src.solvers.solver_factory import SolverFactory
from src.verbosity.verbosity import Verbosity


class Scene:
    _early_stop_checker_factory: EarlyStopCheckerFactory = EarlyStopCheckerFactory()
    _field_factory: FieldFactory = FieldFactory()
    _scheduler_factory: SchedulerFactory = SchedulerFactory()
    _noise_factory: NoiseFactory = NoiseFactory()
    _solver_factory: SolverFactory = SolverFactory()

    def __init__(  # type: ignore
        self,
        *,
        path_to_config: str | None = None,
        config=None,
    ):
        if path_to_config is not None:
            with open(path_to_config, "r") as f:
                config = json.load(f)

        self._answer: Answer = Answer(**config["answer"])

        self._early_stop_checker: EarlyStopCheckerInterface = \
            self._early_stop_checker_factory.construct(config["early_stopping"])

        self._field: FieldInterface = self._field_factory.construct(config["field"])

        self._noise: tp.Optional[noise.NoiseBase] = \
            self._noise_factory.construct(self._answer, config["noise"]) if "noise" in config else None

        self._scheduler: tp.Optional[SchedulerInteface] = \
            self._scheduler_factory.construct(config["scheduler"]) if "scheduler" in config else None

        self._verbosity: Verbosity = Verbosity(**config["verbosity"])

        self._solver: SolverInterface = \
            self._solver_factory.construct(config["solver"], self._field.size, self._field.quality_scale)

        self._n_iterations: int = config["solver"]["params"]["n_iterations"]

        if isinstance(self._solver, swarm.SwarmBase):
            self._solver.correct_positions(self._field.size)

            particles_positions: np.ndarray[tp.Any, np.dtype[np.float64]] = self._solver.get_swarm_positions()

            particles_scores: list[float] = [
                self._field.target_function(*particles_positions[i, :]) for i in range(particles_positions.shape[0])
            ]

            if self._noise is not None:
                if isinstance(self._noise, noise.InverseDistanceNoise):
                    particles_scores = [
                        particles_scores[i] + self._noise.get_noise(particles_positions[i, :])
                        for i in range(len(particles_scores))
                    ]
                elif isinstance(self._noise, noise.RelativeVarianceNoise):
                    particles_scores = [
                        particles_scores[i] + self._noise.get_noise(particles_scores[i])
                        for i in range(len(particles_scores))
                    ]
                else:
                    raise ValueError("noise type type be InverseDistance or RelativeVariance")

            self._solver.update_scores(particles_scores)

        if isinstance(self._solver, grey_wolf_optimization.GreyWolfOptimizationBase):
            self._solver.correct_positions(self._field.size)

            wolves_positions: np.ndarray[tp.Any, np.dtype[np.float64]] = self._solver.get_wolves_positions()

            wolves_scores: list[float] = [
                self._field.target_function(*wolves_positions[i, :]) for i in range(wolves_positions.shape[0])
            ]

            if self._noise is not None:
                if isinstance(self._noise, noise.InverseDistanceNoise):
                    wolves_scores = [
                        wolves_scores[i] + self._noise.get_noise(wolves_positions[i, :])
                        for i in range(len(wolves_scores))
                    ]
                elif isinstance(self._noise, noise.RelativeVarianceNoise):
                    wolves_scores = [
                        wolves_scores[i] + self._noise.get_noise(wolves_scores[i])
                        for i in range(len(wolves_scores))
                    ]
                else:
                    raise ValueError("noise type type be InverseDistance or RelativeVariance")

            self._solver.update_scores(wolves_scores)


        if isinstance(self._solver, artificial_bee_colony.ArtificialBeeColonyBase):
            self._solver.correct_positions(self._field.size)

            employed_bees_positions, onlooker_bees_positions = self._solver.get_bees_positions()

            employed_bees_scores: list[float] = [
                self._field.target_function(*employed_bees_positions[i, :]) for i in range(employed_bees_positions.shape[0])
            ]

            onlooker_bees_scores: list[float] = [
                self._field.target_function(*onlooker_bees_positions[i, :]) for i in range(onlooker_bees_positions.shape[0])
            ]

            if self._noise is not None:
                if isinstance(self._noise, noise.InverseDistanceNoise):
                    employed_bees_scores = [
                        employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i, :])
                        for i in range(len(employed_bees_scores))
                    ]

                    onlooker_bees_scores = [
                        onlooker_bees_scores[i] + self._noise.get_noise(onlooker_bees_positions[i, :])
                        for i in range(len(onlooker_bees_scores))
                    ]
                elif isinstance(self._noise, noise.RelativeVarianceNoise):
                    employed_bees_scores = [
                        employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i])
                        for i in range(len(employed_bees_scores))
                    ]

                    onlooker_bees_scores = [
                        onlooker_bees_scores[i] + self._noise.get_noise(onlooker_bees_scores[i])
                        for i in range(len(onlooker_bees_scores))
                    ]
                else:
                    raise ValueError("noise type type be InverseDistance or RelativeVariance")

            self._solver.update_employed_bees_scores(employed_bees_scores)
            self._solver.correct_positions(self._field.size)
            
            employed_bees_positions, _ = self._solver.get_bees_positions()

            employed_bees_scores: list[float] = [
                self._field.target_function(*employed_bees_positions[i, :]) for i in range(employed_bees_positions.shape[0])
            ]

            if self._noise is not None:
                if isinstance(self._noise, noise.InverseDistanceNoise):
                    employed_bees_scores = [
                        employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i, :])
                        for i in range(len(employed_bees_scores))
                    ]
                elif isinstance(self._noise, noise.RelativeVarianceNoise):
                    employed_bees_scores = [
                        employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i])
                        for i in range(len(employed_bees_scores))
                    ]

            self._solver.updated_employed_bees_scores_only(employed_bees_scores)
            self._solver.update_onlooker_bees_scores(onlooker_bees_scores)

        if self._verbosity.value > 0:
            self._solver.show("Starting position")

    def solve(self) -> tuple[float, float]:
        if isinstance(self._solver, swarm.SwarmBase):
            for iteration_index in range(1, self._n_iterations + 1):
                self._solver.turn()
                self._solver.correct_positions(self._field.size)

                particles_positions: np.ndarray[tp.Any, np.dtype[np.float64]] = self._solver.get_swarm_positions()

                particles_scores: list[float] = [
                    self._field.target_function(*particles_positions[j, :]) for j in range(particles_positions.shape[0])
                ]

                if self._noise is not None:
                    if isinstance(self._noise, noise.InverseDistanceNoise):
                        particles_scores = [
                            particles_scores[i] + self._noise.get_noise(particles_positions[i, :])
                            for i in range(len(particles_scores))
                        ]
                    elif isinstance(self._noise, noise.RelativeVarianceNoise):
                        particles_scores = [
                            particles_scores[i] + self._noise.get_noise(particles_scores[i])
                            for i in range(len(particles_scores))
                        ]

                self._solver.update_scores(particles_scores)

                if self._early_stop_checker.check(self._solver.particles):
                    if self._verbosity.value > 0:
                        self._solver.show("Final Position")

                    return (
                        self._solver.get_position_error(self._answer.answers[0], self._field.size),
                        self._solver.get_path_length(),
                    )

                if self._scheduler is not None:
                    w: float = self._solver.particles[0].w
                    w = self._scheduler.step(w)

                    for j in range(len(self._solver.particles)):
                        self._solver.particles[j].w = w

                if self._verbosity.value > 1:
                    if iteration_index % self._verbosity.period == 0:
                        self._solver.show(f"Epoch #{iteration_index}")

            if self._verbosity.value > 0:
                self._solver.show("Final Position")

            return (
                self._solver.get_position_error(self._answer.answers[0], self._field.size),
                self._solver.get_path_length(),
            )
        if isinstance(self._solver, grey_wolf_optimization.GreyWolfOptimizationBase):
            for iteration_index in range(1, self._n_iterations + 1):
                self._solver.turn()
                self._solver.correct_positions(self._field.size)

                wolves_positions: np.ndarray[tp.Any, np.dtype[np.float64]] = self._solver.get_wolves_positions()

                wolves_scores: list[float] = [
                    self._field.target_function(*wolves_positions[j, :]) for j in range(wolves_positions.shape[0])
                ]

                if self._noise is not None:
                    if isinstance(self._noise, noise.InverseDistanceNoise):
                        wolves_scores = [
                            wolves_scores[i] + self._noise.get_noise(wolves_positions[i, :])
                            for i in range(len(wolves_scores))
                        ]
                    elif isinstance(self._noise, noise.RelativeVarianceNoise):
                        wolves_scores = [
                            wolves_scores[i] + self._noise.get_noise(wolves_scores[i])
                            for i in range(len(wolves_scores))
                        ]

                self._solver.update_scores(wolves_scores)

                if isinstance(self._solver, grey_wolf_optimization.GreyWolfOptimizationImproved):
                    for i in range(len(self._solver.wolves)):
                        self._solver.wolves[i].a = self._solver._a_coef_original * \
                            1 / (1 + np.exp((10 * iteration_index - 5 * self._n_iterations) / self._n_iterations))
                else:
                    for i in range(len(self._solver.wolves)):
                        self._solver.wolves[i].a = self._solver._a_coef_original * (1 - iteration_index / self._n_iterations)

                if self._verbosity.value > 1:
                    if i % self._verbosity.period == 0:
                        self._solver.show(f"Epoch #{i}")

            if self._verbosity.value > 0:
                self._solver.show("Final Position")

            return (
                self._solver.get_position_error(self._answer.answers[0], self._field.size),
                self._solver.get_path_length(),
            )

        if isinstance(self._solver, artificial_bee_colony.ArtificialBeeColonyBase):
            for iteration_index in range(1, self._n_iterations + 1):
                self._solver.turn()
                self._solver.correct_positions(self._field.size)

                employed_bees_positions, onlooker_bees_positions = self._solver.get_bees_positions()

                employed_bees_scores: list[float] = [
                    self._field.target_function(*employed_bees_positions[i, :]) for i in range(employed_bees_positions.shape[0])
                ]

                onlooker_bees_scores: list[float] = [
                    self._field.target_function(*onlooker_bees_positions[i, :]) for i in range(onlooker_bees_positions.shape[0])
                ]

                if self._noise is not None:
                    if isinstance(self._noise, noise.InverseDistanceNoise):
                        employed_bees_scores = [
                            employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i, :])
                            for i in range(len(employed_bees_scores))
                        ]

                        onlooker_bees_scores = [
                            onlooker_bees_scores[i] + self._noise.get_noise(onlooker_bees_positions[i, :])
                            for i in range(len(onlooker_bees_scores))
                        ]
                    elif isinstance(self._noise, noise.RelativeVarianceNoise):
                        employed_bees_scores = [
                            employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i])
                            for i in range(len(employed_bees_scores))
                        ]

                        onlooker_bees_scores = [
                            onlooker_bees_scores[i] + self._noise.get_noise(onlooker_bees_scores[i])
                            for i in range(len(onlooker_bees_scores))
                        ]
                    else:
                        raise ValueError("noise type type be InverseDistance or RelativeVariance")

                self._solver.update_employed_bees_scores(employed_bees_scores)
                self._solver.correct_positions(self._field.size)
                
                employed_bees_positions, _ = self._solver.get_bees_positions()

                employed_bees_scores: list[float] = [
                    self._field.target_function(*employed_bees_positions[i, :]) for i in range(employed_bees_positions.shape[0])
                ]

                if self._noise is not None:
                    if isinstance(self._noise, noise.InverseDistanceNoise):
                        employed_bees_scores = [
                            employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i, :])
                            for i in range(len(employed_bees_scores))
                        ]
                    elif isinstance(self._noise, noise.RelativeVarianceNoise):
                        employed_bees_scores = [
                            employed_bees_scores[i] + self._noise.get_noise(employed_bees_scores[i])
                            for i in range(len(employed_bees_scores))
                        ]

                self._solver.updated_employed_bees_scores_only(employed_bees_scores)
                self._solver.update_onlooker_bees_scores(onlooker_bees_scores)

            return (
                self._solver.get_position_error(self._answer.answers[0], self._field.size),
                self._solver.get_path_length(),
            )

        raise AttributeError("wrong solver type")
