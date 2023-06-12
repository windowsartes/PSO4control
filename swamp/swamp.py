import numpy as np

import sys

# from particle import particle
from field import field

import typing as tp


# just in case I don't know how to properly pre-define class in Python I use this
# '' typing for the Swamp instance
class Particle:
    def __init__(self, field_height: float, field_width: float,
                 field_target_function: tp.Callable[[float, float, tuple[float, float]], float],
                 swamp: 'Swamp'):
        self._position: np.ndarray = np.array([np.random.uniform(0, field_width), np.random.uniform(0, field_height)])
        self._velocity: np.ndarray = np.array((np.random.uniform(-field_width, field_width),
                                               np.random.uniform(-field_height, field_height)))

        self._swamp: 'Swamp' = swamp

        field_target_function = swamp.scene.field.target_function
        field_width = swamp.scene.field.width
        field_height = swamp.scene.field.height

        self._best_score = field_target_function(self._position[0], self._position[1],
                                                 (field_width/2, field_height/2))
        self._best_position = self._position

    @property
    def best_score(self):
        return self._best_score

    @best_score.setter
    def best_score(self, new_score: float):
        self.best_score = new_score

    @property
    def swamp(self):
        return self._swamp

    @property
    def best_position(self):
        return self._best_position

    @best_position.setter
    def best_position(self, new_position: np.ndarray):
        self.best_position = new_position

    def update(self, field_height: float, field_width: float,
               field_target_function: tp.Callable[[float, float, tuple[float, float], float], float]):

        r_personal: float = np.random.uniform()
        r_global: float = np.random.uniform()
        self._velocity = 1 * self._velocity + 2*r_personal*(self._best_position - self._position) + \
                         2*r_global*(self._swamp._best_global_position - self._position)
        self._position = self._velocity + self._position
        current_score = field_target_function(self._position[0], self._position[1],
                                              (field_width/2, field_height/2), 10)
        if current_score > self._best_score:
            self._best_score = current_score
            self._best_position = self._position

        if current_score > self.swamp.best_score:
            self._swamp._best_global_score = current_score
            self._swamp._best_global_position = self._position


class Swamp:
    def __init__(self, n_particles: int, n_iterations):
        self._n_particles = n_particles
        self._n_iterations = n_iterations

        self._best_global_score = sys.float_info.min
        self._best_global_position = None

        self.particles: list[Particle] = []
        for i in range(self._n_particles):
            self.particles.append(Particle(10, 10, field.gaussian, self))
            # self.particles[i].set_swamp(self)

            if self.particles[i].best_score > self._best_global_score:
                self._best_global_score = self.particles[i].best_score
                self._best_global_position = self.particles[i].best_position

        # self.particles = tuple(self.particles)

    def release_the_swamp(self):
        for i in range(self._n_iterations):
            for j in range(len(self.particles)):
                self.particles[j].update(10, 10, field.gaussian)

    def show_best_score(self):
        return self._best_global_score


if __name__ == "__main__":
    my_swamp = Swamp(5, 1000)
    print(my_swamp.show_best_score())
    my_swamp.release_the_swamp()
    print(my_swamp.show_best_score())
