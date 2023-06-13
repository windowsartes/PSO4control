import sys
import typing as tp

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import uniform


# just in case I don't know how to properly pre-define class in Python I use this
# '' typing for the Swamp instance
class Particle:
    def __init__(self, swarm: 'Swarm'):
        self._swarm: 'Swarm' = swarm

        field_target_function = self._swarm.scene.field.target_function
        field_width: float = self._swarm.scene.field.width
        field_height: float = self._swarm.scene.field.height

        if self.swarm.scene.spawn_type == "full location":
            self._position: np.ndarray = np.array([uniform(0, field_width), uniform(0, field_height)])
            self._velocity: np.ndarray = np.array([uniform(-field_width, field_width),
                                                   uniform(-field_height, field_height)])
        elif self.swarm.scene.spawn_type == "edges":
            edge: int = np.random.randint(4)
            if edge == 0:    # left
                self._position: np.ndarray = np.array([0, uniform(0, field_height)])
                self._velocity: np.ndarray = np.array([uniform(0, field_width), uniform(-field_height, field_height)])
            elif edge == 1:  # right
                self._position: np.ndarray = np.array([field_width, uniform(0, field_height)])
                self._velocity: np.ndarray = np.array([uniform(-field_width, 0), uniform(-field_height, field_height)])
            elif edge == 2:  # top
                self._position: np.ndarray = np.array([uniform(0, field_width), 0])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width), uniform(0, field_height)])
            else:            # bottom
                self._position: np.ndarray = np.array([uniform(0, field_width), field_height])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width), uniform(-field_height, 0)])
        elif self.swarm.scene.spawn_type == "small area":
            factor: float = self.swarm.scene.factor
            start_location: np.ndarray = self.swarm.scene.spawn_start_location
            if self.swarm.scene.edge == 0:    # left
                self._position: np.ndarray = np.array([0, uniform(start_location[1] - field_height/factor,
                                                                  start_location[1] + field_height/factor)])
                self._velocity: np.ndarray = np.array([uniform(0, field_width),
                                                       uniform(-field_height, field_height)])
            elif self.swarm.scene.edge == 1:  # right
                self._position: np.ndarray = np.array([field_width, uniform(start_location[1] - field_height/factor,
                                                                            start_location[1] + field_height/factor)])
                self._velocity: np.ndarray = np.array([uniform(-field_width, 0), uniform(-field_height, field_height)])
            elif self.swarm.scene.edge == 2:  # top
                self._position: np.ndarray = np.array([uniform(start_location[0] - field_width/factor,
                                                               start_location[0] + field_width/factor), 0])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width),
                                                       np.random.uniform(0, field_height)])
            else:                             # bottom
                self._position: np.ndarray = np.array([uniform(start_location[0] - field_width/factor,
                                                               start_location[0] + field_width/factor), field_height])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width), uniform(-field_height, 0)])

        self._best_score: float = field_target_function(self._position[0], self._position[1],
                                                        (field_width/2, field_height/2))
        self._best_position: np.ndarray = self._position

    @property
    def best_score(self):
        return self._best_score

    @best_score.setter
    def best_score(self, new_score: float):
        self._best_score = new_score

    @property
    def swarm(self):
        return self._swarm

    @property
    def best_position(self):
        return self._best_position

    @best_position.setter
    def best_position(self, new_position: list | np.ndarray):
        self._best_position = new_position

    @property
    def position(self):
        return self._position

    def update(self):

        field_target_function = self.swarm.scene.field.target_function
        field_width = self.swarm.scene.field.width
        field_height = self.swarm.scene.field.height

        r_personal: float = np.random.uniform()
        r_global: float = np.random.uniform()

        self._velocity = 1 * self._velocity + 2*r_personal*(self._best_position - self._position) + \
                         2*r_global*(self._swarm.best_global_position - self._position)

        self._position = self._velocity + self._position

        current_score = field_target_function(self._position[0], self._position[1],
                                              (field_width/2, field_height/2))

        if current_score > self._best_score:
            self._best_score = current_score
            self._best_position = self._position

        if current_score > self.swarm.best_global_score:
            self._swarm.best_global_score = current_score
            self._swarm.best_global_position = self._position


class Swarm:
    def __init__(self, n_particles: int, n_iterations, scene):
        self._n_particles = n_particles
        self._n_iterations = n_iterations

        self.scene = scene

        self._best_global_score = sys.float_info.min
        self._best_global_position: list[float] | np.ndarray = [0., 0.]

        self.particles: list[Particle] = []
        for i in range(self._n_particles):
            self.particles.append(Particle(self))

            if self.particles[i].best_score > self._best_global_score:
                self._best_global_score = self.particles[i].best_score
                self._best_global_position = self.particles[i].best_position

        coordinates = []
        for i in range(self._n_particles):
            coordinates.append(list(self.particles[i].position))
        coordinates = np.array(coordinates)

        # plt.plot(*zip(*coordinates), marker='o', color='r', ls='', markersize=10)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='o', color='r', ls='', s=20)
        axes = plt.gca()
        axes.set_xlim(0, self.scene.field.width)
        axes.set_ylim(0, self.scene.field.height)
        axes.set_title("Начальное положение")
        plt.show()

    @property
    def best_global_score(self):
        return self._best_global_score

    @best_global_score.setter
    def best_global_score(self, new_value: float):
        self._best_global_score = new_value

    @property
    def best_global_position(self):
        return self._best_global_position

    @best_global_position.setter
    def best_global_position(self, new_position: list[float] | np.ndarray):
        self._best_global_position = new_position

    def release_the_swarm(self):
        for i in range(self._n_iterations):
            for j in range(len(self.particles)):
                self.particles[j].update()
