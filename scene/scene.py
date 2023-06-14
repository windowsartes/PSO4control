import typing as tp
from dataclasses import dataclass

import numpy as np
from numpy.random import uniform

from field import field
from logger import custom_logger
from swarm import swarm


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


class Scene:
    def __init__(self, field_height: float, field_width: float,
                 field_target_function: tp.Callable[[float, float, tuple[float, float]], float],
                 swarm_n_particles: int, swarm_n_iterations: int,
                 spawn_type: str,
                 answer_value: float, answer_location: np.ndarray,
                 verbose: int):
        self.spawn_type = spawn_type

        self.verbose = verbose

        self.answer = Answer(answer_value, answer_location)

        if self.spawn_type == "small_area":
            self.edge: int = np.random.randint(4)
            self.factor: float = 20
            if self.edge == 0:    # left
                self.spawn_start_location: np.ndarray = np.array([0, uniform(field_height/self.factor,
                                                                             field_height-field_height/self.factor)])
            elif self.edge == 1:  # right
                self.spawn_start_location: np.ndarray = np.array([field_width,
                                                                  uniform(field_height/self.factor,
                                                                          field_height - field_height/self.factor)])
            elif self.edge == 2:  # top
                self.spawn_start_location: np.ndarray = np.array([uniform(field_width/self.factor,
                                                                          field_width-field_width/self.factor), 0])
            elif self.edge == 3:  # bottom
                self.spawn_start_location: np.ndarray = np.array([uniform(field_width/self.factor,
                                                                          field_width-field_width/self.factor),
                                                                  field_height])

        self.field: field.Field = field.Field(field_height, field_width, field_target_function)
        self.swarm: swarm.Swarm = swarm.Swarm(swarm_n_particles, swarm_n_iterations, self)

    def run(self) -> tuple[int, float, float, int]:
        # print(self.swarm.best_global_score)
        results = self.swarm.release_the_swarm()
        # print(self.swarm.best_global_score)

        return results


if __name__ == "__main__":

    field_height: float = 10.
    field_width: float = 10.

    n_iterations: int = 500

    verbose: int = 0

    for spawn_type in ["edges", "small_area"]:
        for n_particles in [2, 3, 4, 5, 10, 15, 20]:

            my_logger = custom_logger.CustomLogger("./logs/" + spawn_type + "_" + str(n_particles) + ".csv")

            columns = ["n_iterations", "target_value", "total_path", "exit_code"]

            my_logger.write(columns, "w")

            for i in range(1000):
                my_scene = Scene(field_height, field_width, field.gaussian, n_particles, n_iterations, "small_area",
                                 field.gaussian(field_height/2, field_width/2, (field_height/2, field_width/2)),
                                 np.array([field_height/2, field_width/2]), verbose)
                results = list(my_scene.run())
                my_logger.write(results, "a")
