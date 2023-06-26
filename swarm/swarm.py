import sys
from time import sleep
import os
import typing as tp

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import uniform

from abc import ABC, abstractmethod

from math import ceil


class ParticleInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @property
    @abstractmethod
    def best_score(self):
        pass

    @property
    @abstractmethod
    def best_position(self):
        pass

    @property
    @abstractmethod
    def position(self):
        pass

    @position.setter
    @abstractmethod
    def position(self, new_position: np.ndarray):
        pass

    @property
    @abstractmethod
    def path_length(self):
        pass

    @property
    @abstractmethod
    def velocity(self):
        pass


class ParticleBase(ParticleInterface):
    def __init__(self, swarm: "Swarm"):
        self._swarm: 'Swarm' = swarm

        field_target_function = self._swarm.scene.field.target_function
        field_width: float = self._swarm.scene.field.width
        field_height: float = self._swarm.scene.field.height

        if self._swarm.scene.spawn_type == "full_location":
            self._position: np.ndarray = np.array([uniform(0, field_width), uniform(0, field_height)])
            self._velocity: np.ndarray = np.array([uniform(-field_width, field_width),
                                                   uniform(-field_height, field_height)])
        elif self._swarm.scene.spawn_type == "edges":
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
        elif self._swarm.scene.spawn_type == "small_area":
            factor: float = self._swarm.scene.position_scale
            start_location: np.ndarray = self._swarm.scene.spawn_start_location
            if self._swarm.scene.edge == 0:  # left
                self._position: np.ndarray = np.array([0, uniform(start_location[1] - field_height / factor,
                                                                  start_location[1] + field_height / factor)])
                self._velocity: np.ndarray = np.array([uniform(0, field_width),
                                                       uniform(-field_height, field_height)])
            elif self._swarm.scene.edge == 1:  # right
                self._position: np.ndarray = np.array([field_width, uniform(start_location[1] - field_height / factor,
                                                                            start_location[1] + field_height / factor)])
                self._velocity: np.ndarray = np.array([uniform(-field_width, 0), uniform(-field_height, field_height)])
            elif self._swarm.scene.edge == 2:  # top
                self._position: np.ndarray = np.array([uniform(start_location[0] - field_width / factor,
                                                               start_location[0] + field_width / factor), 0])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width),
                                                       np.random.uniform(0, field_height)])
            else:  # bottom
                self._position: np.ndarray = np.array([uniform(start_location[0] - field_width / factor,
                                                               start_location[0] + field_width / factor), field_height])
                self._velocity: np.ndarray = np.array([uniform(-field_width, field_width), uniform(-field_height, 0)])

        self._velocity /= self._swarm.scene.hyperparameters.velocity_scale
        self._best_score: float = field_target_function(self._position[0], self._position[1],
                                                        (field_width / 2, field_height / 2))
        self._best_position: np.ndarray = self._position

        self._path_length: float = 0

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_position(self) -> np.ndarray:
        return self._best_position

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, new_value: np.ndarray):
        self._position = new_value
        self._correct_position()

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def velocity(self):
        return self._velocity

    def _correct_position(self):
        if self._position[0] < 0:
            self._position[0] = 0
        elif self._position[0] > self._swarm.scene.field.width:
            self._position[0] = self._swarm.scene.field.width

        if self._position[1] < 0:
            self._position[1] = 0
        elif self._position[1] > self._swarm.scene.field.height:
            self._position[1] = self._swarm.scene.field.height


class ParticleCentralized(ParticleBase):
    def __init__(self, swarm: 'SwarmCentralized'):
        super().__init__(swarm)

    def update(self):
        field_target_function = self._swarm.scene.field.target_function
        field_width = self._swarm.scene.field.width
        field_height = self._swarm.scene.field.height

        r_personal: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_scale
        r_global: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_scale

        self._velocity = self._swarm.scene.hyperparameters.w * self._velocity + \
                         self._swarm.scene.hyperparameters.c1 * r_personal * (self._best_position - self._position) + \
                         self._swarm.scene.hyperparameters.c2 * r_global * (self._swarm.best_global_position -
                                                                            self._position)

        self.position = self._velocity + self._position

        self._path_length += np.linalg.norm(self._velocity)

        current_score = field_target_function(self._position[0], self._position[1],
                                              (field_width / 2, field_height / 2))

        if current_score > self._best_score:
            self._best_score = current_score
            self._best_position = self._position

        # self._swarm.get_information_from_particle(self)

        if current_score > self._swarm.best_global_score:
             self._swarm.best_global_score = current_score
             self._swarm.best_global_position = self._position


class ParticleDecentralized(ParticleBase):
    def __init__(self, swarm: 'SwarmDecentralized'):
        super().__init__(swarm)

        self._best_global_score: float = float("-inf")
        self._best_global_position: np.ndarray = self._best_position

    def update(self):
        field_target_function = self._swarm.scene.field.target_function
        field_width = self._swarm.scene.field.width
        field_height = self._swarm.scene.field.height

        r_personal: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_scale
        r_global: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_scale

        self._velocity = self._swarm.scene.hyperparameters.w * self._velocity + \
                         self._swarm.scene.hyperparameters.c1 * r_personal * (self._best_position - self._position) + \
                         self._swarm.scene.hyperparameters.c2 * r_global * (self._best_global_position - self._position)

        self.position = self._velocity + self._position

        self._path_length += np.linalg.norm(self._velocity)

        current_score: float = field_target_function(self._position[0], self._position[1],
                                                     (field_width / 2, field_height / 2))

        if current_score > self._best_score:
            self._best_score = current_score
            self._best_position = self._position

    def update_my_global_information(self):
        for particle in self._swarm._particles:
            if np.linalg.norm(self.position - particle.position) < self._swarm.connection_radius:
                if self._best_global_score < particle.best_score:
                    self._best_global_score = particle.best_score
                    self._best_global_position = particle.best_position


class SwarmInterface(ABC):
    @abstractmethod
    def __init__(self, n_particles: int, n_iterations: int, scene):
        pass

    @abstractmethod
    def get_swarm_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def show_current_position(self, title: str):
        pass

    @abstractmethod
    def release_the_swarm(self) -> tp.Any:
        pass

    @abstractmethod
    def update_global_information(self):
        pass


class SwarmBase(SwarmInterface):
    def __init__(self, n_particles: int, n_iterations: int, scene):
        self._n_particles: int = n_particles
        self._n_iterations: int = n_iterations

        self._scene = scene

        self._best_global_score: float = float("-inf")
        self._best_global_position: np.ndarray = np.array([0., 0.])

        self._particles: list[ParticleBase] = []

    def update_global_information(self):
        for particle in self._particles:
            if self._best_global_score < particle.best_score:
                self._best_global_score = particle.best_score
                self._best_global_position = particle.best_position

    def get_swarm_positions(self) -> np.ndarray:
        print("in method")
        positions: np.ndarray = np.empty((self._n_particles, 2), dtype=np.double)
        for index, particle in enumerate(self._particles):
            print(particle.position)
            positions[index] = particle.position

        return positions

    @property
    def scene(self):
        return self._scene


class SwarmCentralized(SwarmBase):
    def __init__(self, n_particles: int, n_iterations, scene):
        super().__init__(n_particles, n_iterations, scene)

        self._particles: list[ParticleCentralized] = []
        for i in range(self._n_particles):
            self._particles.append(ParticleCentralized(self))

        self.update_global_information()

        if self._scene.verbose > 0:
            self.show_current_position("Начальное положение")

    def get_information_from_particle(self, particle: ParticleCentralized):
        if particle.best_score > self._best_global_score:
            self._best_global_score = particle.best_score
            self._best_global_position = particle.best_position

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
        # print("a")
        early_stopping_small_velocity_count = 0
        early_stopping_small_velocity = False

        early_stopping_around_answer = False

        eps_position = 0.0001
        eps_velocity = 0.0001

        ratio = 0.75

        for i in range(1, self._n_iterations + 1):
            # print("b")
            for j in range(self._n_particles):
                self._particles[j].update()

            for j in range(ceil(ratio * self._n_particles)):
                early_stopping_around_answer_count = 0
                for k in range(self._n_particles):
                    if np.linalg.norm(self._particles[j].position - self._particles[k].position) < eps_position:
                        early_stopping_around_answer_count += 1

                if early_stopping_around_answer_count > ratio * self._n_particles:
                    early_stopping_around_answer = True
                    break

            for j in range(self._n_particles):
                if np.linalg.norm(self._particles[j].velocity) < eps_velocity:
                    early_stopping_small_velocity_count += 1

            if early_stopping_around_answer:
                break

            elif early_stopping_small_velocity_count > ratio * self._n_particles:
                early_stopping_small_velocity = True
                break

            if i % 1 == 0:
                if self.scene.verbose > 1:
                    self.show_current_position(str(i))
                    # os.system('clear')

            early_stopping_small_velocity_count = 0

            self.scene.inertia_scheduler.step()

        total_path_length: float = 0
        for j in range(self._n_particles):
            total_path_length += self._particles[j].path_length

        result = [i, self.scene.answer.value - self.best_global_score,
                  (self.scene.answer.value - self.best_global_score) / self.scene.answer.value, total_path_length,
                  total_path_length/self._n_particles]

        if early_stopping_around_answer:
            result = result + [1]
        elif early_stopping_small_velocity:
            result = result + [2]
        else:
            result = result + [0]

        if self.scene.verbose > 0:
            self.show_current_position("Последняя итерация")

        return tuple(result)


class SwarmDecentralized(SwarmBase):
    def __init__(self, n_particles: int, n_iterations: int, connection_radius: float, scene):
        super().__init__(n_particles, n_iterations, scene)

        self.connection_radius = connection_radius

        # self._particles: list[ParticleDecentralized] = []
        self._particles: list[ParticleDecentralized] = []
        for i in range(self._n_particles):
            self._particles.append(ParticleDecentralized(self))

        self.update_global_information()

        plt.ion()

        if self._scene.verbose > 0:
            self.show_current_position("Начальное положение")

    def show_current_position(self, title: str):
        coordinates: np.ndarray = self.get_swarm_positions()

        plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='o', color='r', ls='', s=20)
        axes = plt.gca()
        axes.set_xlim(0, self.scene.field.width)
        axes.set_ylim(0, self.scene.field.height)
        axes.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            axes.annotate(label, (coordinates[index][0], coordinates[index][1]))

        for coordinate in coordinates:
            circle = plt.Circle(coordinate, self.connection_radius, color="g", fill=False, linestyle="--")
            axes.add_patch(circle)

        # plt.show()

        plt.pause(0.1)
        plt.clf()

    def _get_best_global_score(self, update=True) -> float:
        if update:
            self.update_global_information()

        return self._best_global_score

    def _get_best_global_position(self, update=True) -> np.ndarray:
        if update:
            self.update_global_information()

        return self._best_global_position

    def release_the_swarm(self) -> tuple[int, float, float, int]:
        early_stopping_small_velocity_count = 0
        early_stopping_small_velocity = False

        early_stopping_around_answer = False

        eps_position = 0.001
        eps_velocity = 0.001

        ratio = 0.75

        for i in range(1, self._n_iterations + 1):
            for j in range(self._n_particles):
                self._particles[j].update()

            for j in range(self._n_particles):
                self._particles[j].update_my_global_information()

            for j in range(ceil(ratio*self._n_particles)):
                early_stopping_around_answer_count = 0
                for k in range(self._n_particles):
                    if np.linalg.norm(self._particles[j].position - self._particles[k].position) < eps_position:
                        early_stopping_around_answer_count += 1

                if early_stopping_around_answer_count > ratio * self._n_particles:
                    early_stopping_around_answer = True
                    break

            for j in range(self._n_particles):
                if np.linalg.norm(self._particles[j].velocity) < eps_velocity:
                    early_stopping_small_velocity_count += 1

            if early_stopping_around_answer:
                break

            elif early_stopping_small_velocity_count > ratio * self._n_particles:
                early_stopping_small_velocity = True
                break

            if i % 1 == 0:
                if self._scene.verbose > 1:
                    self.show_current_position(str(i))
                    # os.system('clear')

            early_stopping_small_velocity_count = 0

            self._scene.inertia_scheduler.step()

        total_path_length: float = 0
        for j in range(self._n_particles):
            total_path_length += self._particles[j].path_length

        result = [i,
                  self._scene.answer.value - self._get_best_global_score(),
                  (self._scene.answer.value - self._get_best_global_score(False)) / self._scene.answer.value,
                  total_path_length,
                  total_path_length/self._n_particles]

        if early_stopping_around_answer:
            result = result + [1]
        elif early_stopping_small_velocity:
            result = result + [2]
        else:
            result = result + [0]

        if self._scene.verbose > 0:
            self.show_current_position("Последняя итерация")

        return tuple(result)
