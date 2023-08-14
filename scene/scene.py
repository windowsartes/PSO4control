import typing as tp
import json
from dataclasses import dataclass

import numpy as np
from numpy.random import uniform

from field import field as fl
from logger import custom_logger
from swarm import swarm as sw

from gradient_methods import gradient_methods


class InertiaScheduler:
    def __init__(self, step_size, gamma, scene: 'Scene'):
        self._step_size: int = step_size
        self._gamma: float = gamma
        self._steps: int = 0
        self._scene = scene

    def step(self):
        self._steps += 1
        if self._steps % self._step_size == 0:
            self._scene.hyperparameters.w *= self._gamma


@dataclass
class Answer:
    _value: float
    _position: np.ndarray

    @property
    def value(self):
        return self._value

    @property
    def position(self):
        return self._position


@dataclass
class Hyperparameters:
    _w: float
    _c1: float
    _c2: float
    _velocity_scale: float
    _inertia_scheduler_step_size: int
    _inertia_scheduler_gamma: float
    _connect_radius: float = float("inf")
    _noise_scale: float = 0.001425

    @property
    def noise_scale(self):
        return self._noise_scale

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, new_value):
        self._w = new_value

    @property
    def c1(self) -> float:
        return self._c1

    @c1.setter
    def c1(self, new_value):
        self._c1 = new_value

    @property
    def c2(self) -> float:
        return self._c2

    @c2.setter
    def c2(self, new_value):
        self._c2 = new_value

    @property
    def velocity_scale(self) -> float:
        return self._velocity_scale

    @property
    def inertia_scheduler_step_size(self) -> int:
        return self._inertia_scheduler_step_size

    @inertia_scheduler_step_size.setter
    def inertia_scheduler_step_size(self, new_value: int):
        self._inertia_scheduler_step_size = new_value

    @property
    def inertia_scheduler_gamma(self) -> float:
        return self._inertia_scheduler_gamma

    @inertia_scheduler_gamma.setter
    def inertia_scheduler_gamma(self, new_value: float):
        self._inertia_scheduler_gamma = new_value


class Scene:
    def __init__(self,
                 # field_height: float, field_width: float,
                 # field_target_function: tp.Callable[[float, float, tuple[float, float]], float],
                 swarm_n_particles: int, swarm_n_iterations: int,
                 spawn_type: str,
                 # answer_value: float, answer_location: np.ndarray,
                 verbose: int,
                 answer: Answer,
                 field: fl.GaussianField,
                 hyperparameters: Hyperparameters,
                 swarm_type: str = "centralized"):
        self.answer: Answer = answer
        self.field: fl.FieldInterface = field
        self.hyperparameters: Hyperparameters = hyperparameters
        self.verbose: int = verbose
        self.spawn_type: str = spawn_type

        self.inertia_scheduler = InertiaScheduler(self.hyperparameters.inertia_scheduler_step_size,
                                                  self.hyperparameters.inertia_scheduler_gamma,
                                                  self)

        if self.spawn_type == "small_area":
            self.edge: int = np.random.randint(4)
            self.position_scale: float = 20
            if self.edge == 0:    # left
                self.spawn_start_location: np.ndarray = np.array([0, uniform(self.field.height/self.position_scale,
                                                                             self.field.height -
                                                                             self.field.height/self.position_scale)])
            elif self.edge == 1:  # right
                self.spawn_start_location: np.ndarray = np.array([self.field.width,
                                                                  uniform(self.field.height/self.position_scale,
                                                                          self.field.height -
                                                                          self.field.height/self.position_scale)])
            elif self.edge == 2:  # top
                self.spawn_start_location: np.ndarray = np.array([uniform(self.field.width/self.position_scale,
                                                                          self.field.width -
                                                                          self.field.width/self.position_scale), 0])
            elif self.edge == 3:  # bottom
                self.spawn_start_location: np.ndarray = np.array([uniform(self.field.width/self.position_scale,
                                                                          self.field.width -
                                                                          self.field.width/self.position_scale),
                                                                  self.field.height])
        if swarm_type == "centralized":
            self.swarm: sw.SwarmBase = sw.SwarmCentralized(swarm_n_particles, swarm_n_iterations, self)
        elif swarm_type == "decentralized":
            self.swarm: sw.SwarmBase = sw.SwarmDecentralized(swarm_n_particles, swarm_n_iterations,
                                                             self.hyperparameters._connect_radius, self)
        elif swarm_type == "corrupted":
            self.swarm: sw.SwarmCorrupted = sw.SwarmCorrupted(swarm_n_particles, swarm_n_iterations,
                                                              self.hyperparameters._connect_radius, self)

    def run(self) -> tuple[int, float, float, int]:
        # print(self.swarm.best_global_score)
        results = self.swarm.release_the_swarm()
        # print(self.swarm.best_global_score)

        return results


class SceneGrad:
    def __init__(self, n_iterations, answer: Answer,
                 field: fl.GaussianField, verbose):
        self.answer = answer
        self.field = field
        self.verbose = verbose

        self.solver = gradient_methods.GradientLift(self, n_iterations)

    def run(self):
        results = self.solver.run()
        return results


if __name__ == "__main__":
    json_config = """
    {
    "field":
        {
        "target_function": "gaussian",
        "height": 10,
        "width": 10,
        "correctness_scale": 100
         },
    "answer": [5, 5],
    "solver":
        {
        "type": "decentralized",
        "spawn_type": "small_area",
        "n_iterations": 1000,
        "n_particles": 10,
        "hyperparams":
            {
            "coefficients":
                {
                "w": 1,
                "c1": 2,
                "c2": 2
                },
            "inertia_scheduler":
                {
                "gamma": 0.75,
                "step_size": 100
                }    
            },
            "early_stopping":
                {
                "around_point": 
                    {
                    "epsilon": 0.0001,
                    "ratio": 0.75
                    },
                "velocity":
                    {
                    "epsilon: 0.0001,
                    "ratio": 0.75
                    }
                },
            }
            "connection_radius": 0.1,
            "velocity_factor": 50,
        },
    "verbose": 2,    
    """

    height = 10.
    width = 10.

    answer = Answer(fl.gaussian(height/2, width/2, (height/2, width/2)), np.array([height/2, width/2]))
    field = fl.GaussianField(height, width, fl.gaussian, fl.gaussian_symbolic)
    hyperparams = Hyperparameters(_w=1, _c1=2, _c2=2, _velocity_scale=50,
                                  _inertia_scheduler_step_size=100, _inertia_scheduler_gamma=0.75,
                                  _connect_radius=height*0.1)

    my_scene = Scene(20, 1000, "small_area", 2, answer, field, hyperparams, "centralized")
    _ = my_scene.run()
    print(_)

    """
    height = 10.
    width = 10.

    answer = Answer(fl.gaussian(height / 2, width / 2), np.array([height / 2, width / 2]))
    field = fl.Field(height, width, fl.gaussian)

    my_logger = custom_logger.CustomLogger("./logs/gradient/gradient_lift.csv")
    # columns = ["n_iterations", "absolute_error", "relative_error", "total_path"]
    # my_logger.write(columns, "w")

    for i in range(1000-943):
        my_scene = SceneGrad(3000, answer, field, 0)
        results = list(my_scene.run())
        results = results
        my_logger.write(results, "a")
    """
