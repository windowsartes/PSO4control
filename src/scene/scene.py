import json
import os
import pathlib
import typing as tp

import numpy as np
from tqdm import tqdm

from src.answer.answer import Answer
from src.early_stopping.checker import EarlyStopCheckerInterface
from src.early_stopping.checker_factory import EarlyStopCheckerFactory
from src.field.field import FieldInterface
from src.field.field_factory import FieldFactory
from src.noise import noise
from src.noise.noise_factory import NoiseFactory
from src.scheduler.scheduler import SchedulerInteface
from src.scheduler.scheduler_factory import SchedulerFactory
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

    def __init__(
        self,
        *,
        path_to_config: str | None = None,
        config = None,
    ):
        if path_to_config is not None:
            with open(path_to_config, "r") as f:
                config = json.load(f)

        self._answer: Answer = Answer(**config["answer"])

        self._early_stop_checker: EarlyStopCheckerInterface = \
            self._early_stop_checker_factory.construct(config["early_stopping"])

        self._field: FieldInterface = self._field_factory.construct(config["field"])

        field_dump_dir: pathlib.Path = pathlib.Path("./stored_field")
        field_dump_dir.mkdir(parents=True, exist_ok=True)

        if not os.path.isfile(f"{str(field_dump_dir)}/field.pickle"):
            self._field.compute_and_save_field(f"{str(field_dump_dir)}/field.pickle")

        # self._field.show()
        '''
        total = 0.
        for x in tqdm(np.arange(0, self._field.size, 0.001)):
            for y in np.arange(0, self._field.size, 0.001):
                add_is_bigger = self._field.check_additional(x, y)
                total += add_is_bigger

        print(total / (self._field.size * 1000 * self._field.size * 1000))
        '''
        total = 0.

        for x in tqdm(np.arange(0, self._field.size, 0.001)):
            for y in [0., ]:
                add_is_bigger = self._field.check_additional(x, y)
                total += add_is_bigger
        print(total / (self._field.size * 1000) / 4)
    
        '''
        for y in tqdm(np.arange(0, self._field.size, 0.001)):
            for x in [0.0, 10.0]:
                add_is_bigger = self._field.check_additional(x, y)
                total += add_is_bigger
        '''
        
        # print(total / (self._field.size * 1000))

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

        if self._verbosity.value > 0:
            self._solver.show("Starting position")

    def solve(self) -> None:
        if isinstance(self._solver, swarm.SwarmBase):
            for i in range(1, self._n_iterations + 1):
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
                    if i % self._verbosity.period == 0:
                        self._solver.show(f"Epoch #{i}")

            if self._verbosity.value > 0:
                self._solver.show("Final Position")

            return (
                self._solver.get_position_error(self._answer.answers[0], self._field.size),
                self._solver.get_path_length(),
            )

        """
        if isinstance(self._solver, gradient.GradientLift):
            for i in range(1, self._n_iterations + 1):
                self._solver.turn(self._field.gradient(*self._solver.position))

                if self._verbosity.value > 1:
                    if i % self._verbosity.period == 0:
                        self._solver.show(f"Epoch #{i}")

            self._solver.show("Final Position")

        if isinstance(self._solver, gradient.NewtonsMethod):
            for i in range(1, self._n_iterations + 1):
                self._solver.turn(
                    self._field.gradient(*self._solver.position),
                    self._field.hessian(*self._solver.position),
                )

                if self._verbosity.value > 1:
                    if i % self._verbosity.period == 0:
                        self._solver.show(f"Epoch #{i}")

            self._solver.show("Final Position")
        """
