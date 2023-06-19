import typing as tp
from dataclasses import dataclass

import numpy as np
from numpy.random import uniform

from field import field as fl
from logger import custom_logger
from swarm import swarm as sw


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
            # print("w = ", self._scene.hyperparameters.w)


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
    _speed_scale: float
    _inertia_scheduler_step_size: int
    _inertia_scheduler_gamma: float

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
    def speed_scale(self) -> float:
        return self._speed_scale

    #@speed_scale.setter
    #def speed_scale(self, new_value):
    #    self._speed_scale = new_value

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
                 field: fl.Field,
                 hyperparameters: Hyperparameters):
        self.answer: Answer = answer
        self.field: fl.Field = field
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

        self.swarm: sw.Swarm = sw.Swarm(swarm_n_particles, swarm_n_iterations, self)

    def run(self) -> tuple[int, float, float, int]:
        print(self.swarm.best_global_score)
        results = self.swarm.release_the_swarm()
        print(self.swarm.best_global_score)

        return results


if __name__ == "__main__":
    """
    field_height: float = 10.
    field_width: float = 10.

    n_iterations: int = 500

    verbose: int = 0

    for spawn_type in ["edges", "small_area"]:
        for n_particles in [2, 3, 4, 5, 10, 15, 20]:

            my_logger = custom_logger.CustomLogger("./logs/" + spawn_type + "_" + str(n_particles) + ".csv")

            columns = ["n_iterations", "absolute_error", "relative_error", "total_path", "exit_code", "spawn_type"]

            my_logger.write(columns, "w")

            for i in range(100):
                my_scene = Scene(field_height, field_width, field.gaussian, n_particles, n_iterations, "small_area",
                                 field.gaussian(field_height/2, field_width/2, (field_height/2, field_width/2)),
                                 np.array([field_height/2, field_width/2]), verbose)
                results = list(my_scene.run())
                results = results + [spawn_type]
                my_logger.write(results, "a")
    """
    height = 10
    width = 10

    answer = Answer(fl.gaussian(height/2, width/2, (height/2, width/2)), np.array([height/2, width/2]))
    field = fl.Field(height, width, fl.gaussian)
    hyperparams = Hyperparameters(_w=1, _c1=2/50, _c2=2/50, _speed_scale=50,
                                  _inertia_scheduler_step_size=100, _inertia_scheduler_gamma=0.75)

    my_scene = Scene(5, 1000, "small_area", 1, answer, field, hyperparams)
    _ = my_scene.run()
    print(_)
