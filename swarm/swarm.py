import sys
from time import sleep
import os
import typing as tp

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import uniform


# just in case I don't know how to properly pre-define class in Python I use this
# '' typing for the Swarm instance
class Particle:
    def __init__(self, swarm: 'Swarm'):
        self._swarm: 'Swarm' = swarm

        field_target_function = self._swarm.scene.field.target_function
        field_width: float = self._swarm.scene.field.width
        field_height: float = self._swarm.scene.field.height

        if self.swarm.scene.spawn_type == "full_location":
            self._position: np.ndarray = np.array([uniform(0, field_width), uniform(0, field_height)])
            self._velocity: np.ndarray = np.array([uniform(-field_width, field_width),
                                                   uniform(-field_height, field_height)])
        elif self.swarm.scene.spawn_type == "edges":
            edge: int = np.random.randint(4)
            if edge == 0:  # left
                self._position: np.ndarray = np.array([0, uniform(0, field_height)])
                self._velocity: np.ndarray = np.array([uniform(0, field_width), uniform(-field_height, field_height)])
            elif edge == 1:  # right
                self._position: np.ndarray = np.array([field_width, uniform(0, field_height)])
                self._velocity: np.ndarray = np.array([uniform(-field_width, 0), uniform(-field_height, field_height)])
            elif edge == 2:  # top
                self._position: np.ndarray = np.array([uniform(0, field_width), 0])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width), uniform(0, field_height)])
            else:  # bottom
                self._position: np.ndarray = np.array([uniform(0, field_width), field_height])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width), uniform(-field_height, 0)])
        elif self.swarm.scene.spawn_type == "small_area":
            factor: float = self.swarm.scene.position_scale
            start_location: np.ndarray = self.swarm.scene.spawn_start_location
            if self.swarm.scene.edge == 0:  # left
                self._position: np.ndarray = np.array([0, uniform(start_location[1] - field_height / factor,
                                                                  start_location[1] + field_height / factor)])
                self._velocity: np.ndarray = np.array([uniform(0, field_width),
                                                       uniform(-field_height, field_height)])
            elif self.swarm.scene.edge == 1:  # right
                self._position: np.ndarray = np.array([field_width, uniform(start_location[1] - field_height / factor,
                                                                            start_location[1] + field_height / factor)])
                self._velocity: np.ndarray = np.array([uniform(-field_width, 0), uniform(-field_height, field_height)])
            elif self.swarm.scene.edge == 2:  # top
                self._position: np.ndarray = np.array([uniform(start_location[0] - field_width / factor,
                                                               start_location[0] + field_width / factor), 0])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width),
                                                       np.random.uniform(0, field_height)])
            else:  # bottom
                self._position: np.ndarray = np.array([uniform(start_location[0] - field_width / factor,
                                                               start_location[0] + field_width / factor), field_height])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width), uniform(-field_height, 0)])

        self._velocity /= self._swarm.scene.hyperparameters.speed_scale
        self._best_score: float = field_target_function(self._position[0], self._position[1],
                                                        (field_width / 2, field_height / 2))
        self._best_position: np.ndarray = self._position
        self._path_length: float = 0

    @property
    def best_score(self) -> float:
        return self._best_score

    @best_score.setter
    def best_score(self, new_score: float):
        self._best_score = new_score

    @property
    def best_position(self) -> np.ndarray:
        return self._best_position

    @best_position.setter
    def best_position(self, new_position: list | np.ndarray):
        self._best_position = new_position

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, new_value: np.ndarray):
        self._position = new_value
        self.correct_position()

    @property
    def path_length(self) -> float:
        return self._path_length

    @path_length.setter
    def path_length(self, new_value: float):
        self._path_length = new_value

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, new_value: np.ndarray):
        self._velocity = new_value

    @property
    def swarm(self):
        return self._swarm

    def correct_position(self):
        if self._position[0] < 0:
            self._position[0] = 0
        elif self._position[0] > self._swarm.scene.field.width:
            self._position[0] = self._swarm.scene.field.width

        if self._position[1] < 0:
            self._position[1] = 0
        elif self._position[1] > self._swarm.scene.field.height:
            self._position[1] = self._swarm.scene.field.height

    def update(self):
        field_target_function = self.swarm.scene.field.target_function
        field_width = self.swarm.scene.field.width
        field_height = self.swarm.scene.field.height

        r_personal: float = np.random.uniform()
        r_global: float = np.random.uniform()

        self._velocity = self.swarm.scene.hyperparameters.w * self._velocity + \
                         self.swarm.scene.hyperparameters.c1 * r_personal * (self.best_position - self.position) + \
                         self.swarm.scene.hyperparameters.c2 * r_global * (self.swarm.best_global_position -
                                                                           self.position)

        self.position = self.velocity + self.position

        self.path_length += np.linalg.norm(self.velocity)

        current_score = field_target_function(self.position[0], self.position[1],
                                              (field_width / 2, field_height / 2))

        if current_score > self._best_score:
            self.best_score = current_score
            self.best_position = self.position

        if current_score > self.swarm.best_global_score:
            self.swarm.best_global_score = current_score
            self.swarm.best_global_position = self.position


class Swarm:
    def __init__(self, n_particles: int, n_iterations, scene):
        self._n_particles = n_particles
        self._n_iterations = n_iterations

        self._scene = scene

        self._best_global_score = sys.float_info.min
        self._best_global_position: list[float] | np.ndarray = [0., 0.]

        self.particles: list[Particle] = []
        for i in range(self._n_particles):
            self.particles.append(Particle(self))

            if self.particles[i].best_score > self.best_global_score:
                self.best_global_score = self.particles[i].best_score
                self.best_global_position = self.particles[i].best_position

        if self._scene.verbose > 0:
            self.show_current_position("Начальное положение")

    @property
    def best_global_score(self):
        return self._best_global_score

    @best_global_score.setter
    def best_global_score(self, new_value: float):
        self._best_global_score = new_value

    @property
    def best_global_position(self) -> np.ndarray:
        return self._best_global_position

    @best_global_position.setter
    def best_global_position(self, new_position: np.ndarray):
        self._best_global_position = new_position

    @property
    def scene(self):
        return self._scene

    def get_swarm_positions(self) -> np.ndarray:
        positions: np.ndarray = np.empty((len(self.particles), 2), dtype=np.double)
        for index, particle in enumerate(self.particles):
            positions[index] = particle.position

        return positions

    def show_current_position(self, title: str):
        coordinates: np.ndarray = self.get_swarm_positions()

        plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='o', color='r', ls='', s=20)
        axes = plt.gca()
        axes.set_xlim(0, self.scene.field.width)
        axes.set_ylim(0, self.scene.field.height)
        axes.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            axes.annotate(label, (coordinates[index][0], coordinates[index][1]))

        plt.show()

        sleep(1)
        plt.close(plt.gcf())

    def release_the_swarm(self) -> tuple[int, float, float, int]:
        early_stopping_small_velocity_count = 0
        early_stopping_small_velocity = False

        early_stopping_around_answer = False

        eps_position = 0.0001
        eps_velocity = 0.0001

        ratio = 0.75

        for i in range(1, self._n_iterations + 1):
            for j in range(self._n_particles):
                self.particles[j].update()

            for j in range(self._n_particles):
                early_stopping_around_answer_count = 0
                for k in range(self._n_particles):
                    if np.linalg.norm(self.particles[j].position - self.particles[k].position) < eps_position:
                        early_stopping_around_answer_count += 1

                if early_stopping_around_answer_count > ratio * self._n_particles:
                    early_stopping_around_answer = True
                    break
                if np.linalg.norm(self.particles[j].velocity) < eps_velocity:
                    early_stopping_small_velocity_count += 1

            if early_stopping_around_answer:
                break

            elif early_stopping_small_velocity_count > ratio * self._n_particles:
                early_stopping_small_velocity = True
                break

            for j in range(self._n_particles):
                print(self.particles[j].velocity)

            if i % 1 == 0:
                if self.scene.verbose > 1:
                    self.show_current_position(str(i))
                    # os.system('clear')

            early_stopping_small_velocity_count = 0

            self.scene.inertia_scheduler.step()

        total_path_length: float = 0
        for j in range(self._n_particles):
            total_path_length += self.particles[j].path_length

        result = [i, self.scene.answer.value - self.best_global_score,
                  (self.scene.answer.value - self.best_global_score) / self.scene.answer.value, total_path_length,
                  total_path_length/self._n_particles]

        if early_stopping_around_answer:
            result = result + [1]
        elif early_stopping_small_velocity:
            result = result + [2]
        else:
            result = result + [0]

        self.show_current_position("Последняя итерация")

        return tuple(result)
