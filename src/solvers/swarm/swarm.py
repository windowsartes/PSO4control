import os
import typing as tp
import pickle
from abc import ABC, abstractmethod
from math import ceil

import matplotlib
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from numpy.random import uniform

from src.solvers.solver_interface import SolverInterface


class ParticleInterface(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        """
        One of the most import methods to implement; Used for update particle's position and velocity attributes;

        Returns: Nothing;
        """
        pass

    @property
    @abstractmethod
    def best_score(self) -> float:
        """
        Returns best score that particle has achieved during training;

        Returns: particle's best score;
        """
        pass

    @property
    @abstractmethod
    def best_position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        """
        Returns best position that particle has found during training;

        Returns: particle's best found position;
        """
        pass

    @property
    @abstractmethod
    def position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        """
        Returns particles current position;

        Returns: particle's current position as a np.ndarray;
        """
        pass

    @position.setter
    @abstractmethod
    def position(self, new_position: np.ndarray[tp.Any, np.dtype[np.float64]]) -> None:
        """
        A positions setter; using it, you can properly change particle's position outside the class;
        Args:
            new_position: new particle's position as a nd.ndarray;

        Returns: Nothing;
        """
        pass

    @property
    @abstractmethod
    def path_length(self) -> float:
        pass

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass


class ParticleBase(ParticleInterface):
    def __init__(self, swarm: 'SwarmBase'):
        self._swarm: 'SwarmBase' = swarm

        field_width: float = self._swarm.scene.field.width
        field_height: float = self._swarm.scene.field.height

        self._position: np.ndarray[tp.Any, np.dtype[np.float64]]
        self._velocity: np.ndarray[tp.Any, np.dtype[np.float64]]

        if self._swarm.scene.spawn_type == "full_location":
            self._position = np.array([uniform(0, field_width), uniform(0, field_height)])
            self._velocity = np.array([uniform(-field_width, field_width), uniform(-field_height, field_height)])
        elif self._swarm.scene.spawn_type == "edges":
            edge: int = np.random.randint(4)
            if edge == 0:  # left
                self._position = np.array([0, uniform(0, field_height)])
                self._velocity = np.array([uniform(0, field_width), uniform(-field_height, field_height)])
            elif edge == 1:  # right
                self._position = np.array([field_width, uniform(0, field_height)])
                self._velocity = np.array([uniform(-field_width, 0), uniform(-field_height, field_height)])
            elif edge == 2:  # top
                self._position = np.array([uniform(0, field_width), 0])
                self._velocity = np.array([uniform(-field_width, field_width), uniform(0, field_height)])
            else:  # bottom
                self._position = np.array([uniform(0, field_width), field_height])
                self._velocity = np.array([uniform(-field_width, field_width), uniform(-field_height, 0)])
        elif self._swarm.scene.spawn_type == "small_area":
            factor: float = self._swarm.scene.hyperparameters.position_factor
            start_location: np.ndarray[tp.Any, np.dtype[np.float64]] = self._swarm.scene.spawn_start_location
            if self._swarm.scene.edge == 0:  # left
                self._position = np.array([0, uniform(start_location[1] - field_height / factor,
                                                      start_location[1] + field_height / factor)])
                self._velocity = np.array([uniform(0, field_width), uniform(-field_height, field_height)])
            elif self._swarm.scene.edge == 1:  # right
                self._position = np.array([field_width, uniform(start_location[1] - field_height / factor,
                                                                start_location[1] + field_height / factor)])
                self._velocity = np.array([uniform(-field_width, 0), uniform(-field_height, field_height)])
            elif self._swarm.scene.edge == 2:  # top
                self._position = np.array([uniform(start_location[0] - field_width / factor,
                                                   start_location[0] + field_width / factor), 0])
                self._velocity = np.array([uniform(-field_width, field_width), np.random.uniform(0, field_height)])
            else:  # bottom
                self._position = np.array([uniform(start_location[0] - field_width / factor,
                                                   start_location[0] + field_width / factor), field_height])
                self._velocity = np.array([uniform(-field_width, field_width), uniform(-field_height, 0)])

        self._velocity /= self._swarm.scene.hyperparameters.velocity_factor
        self._best_score: float = self._swarm.scene.field.target_function(*list(self._position))
        self._best_position: np.ndarray[tp.Any, np.dtype[np.float64]] = self._position

        self._path_length: float = 0

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._best_position

    @property
    def position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._position

    @position.setter
    def position(self, new_value: np.ndarray[tp.Any, np.dtype[np.float64]]) -> None:
        self._position = new_value
        self._correct_position()

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def velocity(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._velocity

    def _correct_position(self) -> None:
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
        self._swarm: 'SwarmCentralized' = swarm

    def update(self) -> None:
        r_personal: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_factor
        r_global: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_factor

        self._velocity = self._swarm.scene.hyperparameters.w * self._velocity + \
            self._swarm.scene.hyperparameters.c1 * r_personal * (self._best_position - self._position) + \
            self._swarm.scene.hyperparameters.c2 * r_global * (self._swarm.best_global_position - self._position)

        self.position = self._velocity + self._position

        self._path_length += float(np.linalg.norm(self._velocity))

        current_score = self._swarm.scene.field.target_function(*list(self._position))

        if current_score > self._best_score:
            self._best_score = current_score
            self._best_position = self._position

        if current_score > self._swarm.best_global_score:
            self._swarm.best_global_score = current_score
            self._swarm.best_global_position = self._position


class ParticleDecentralizedBase(ParticleBase):
    def __init__(self, swarm: 'SwarmDecentralizedBase') -> None:
        super().__init__(swarm)
        self._swarm: SwarmDecentralizedBase = swarm

        self._best_global_score: float = float("-inf")
        self._best_global_position: np.ndarray[tp.Any, np.dtype[np.float64]] = self._best_position

    def update(self) -> None:
        r_personal: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_factor
        r_global: float = np.random.uniform()/self._swarm.scene.hyperparameters.velocity_factor

        self._velocity = self._swarm.scene.hyperparameters.w * self._velocity + \
            self._swarm.scene.hyperparameters.c1 * r_personal * (self._best_position - self._position) + \
            self._swarm.scene.hyperparameters.c2 * r_global * (self._best_global_position - self._position)

        self.position = self._velocity + self._position

        self._path_length += float(np.linalg.norm(self._velocity))

    def update_my_global_information(self) -> None:
        """
        This method is used when decentralized particle want's to update its global information: best found score ant
        its position. Note that only particle, that located close enough to target particle can share information;

        Returns: Nothing
        """
        for particle in self._swarm.particles:
            if np.linalg.norm(self.position - particle.position) < \
                    self._swarm.connection_radius*self._swarm.scene.field.height:
                if self._best_global_score < particle.best_score:
                    self._best_global_score = particle.best_score
                    self._best_global_position = particle.best_position


class ParticleDecentralized(ParticleDecentralizedBase):
    def __init__(self, swarm: 'SwarmDecentralized'):
        super().__init__(swarm)
        self._swarm: SwarmDecentralized = swarm

    def update(self) -> None:
        super().update()

        current_score: float = self._swarm.scene.field.target_function(self._position[0], self._position[1])
        if current_score > self._best_score:
            self._best_score = current_score
            self._best_position = self._position


class ParticleCorrupted(ParticleDecentralizedBase):
    def __init__(self, swarm: 'SwarmCorrupted'):
        super().__init__(swarm)
        self._swarm: SwarmCorrupted = swarm
        self._best_score += self._swarm.scene.noise.add_noise(self._position, self._swarm.scene.answer.position)

    def update(self) -> None:
        super().update()

        current_score: float = self._swarm.scene.field.target_function(*list(self._position)) + \
            self._swarm.scene.noise.add_noise(self._position, self._swarm.scene.answer.position)

        if current_score > self._best_score:
            self._best_score = current_score
            self._best_position = self._position


class SwarmInterface(SolverInterface):
    @abstractmethod
    def __init__(self, n_particles: int, n_iterations: int, scene) -> None:  # type: ignore
        pass

    @abstractmethod
    def get_swarm_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def show_current_position(self, title: str) -> None:
        pass

    @abstractmethod
    def run(self) -> tp.Any:
        pass

    @abstractmethod
    def update_global_information(self) -> None:
        pass


class SwarmBase(SwarmInterface):
    def __init__(self, n_particles: int, n_iterations: int, scene):  # type: ignore
        self._n_particles: int = n_particles
        self._n_iterations: int = n_iterations

        self._scene = scene

        self._best_global_score: float = float("-inf")
        self._best_global_position: np.ndarray[tp.Any, np.dtype[np.float64]] = np.array([0., 0.])

        self._particles: list[ParticleBase] = []

        if self._scene.verbosity.value > 0:
            if not os.path.isfile("./stored_field/field.pickle"):
                self.scene.field.compute_and_save_field()

    def update_global_information(self) -> None:
        for particle in self._particles:
            if self._best_global_score < particle.best_score:
                self._best_global_score = particle.best_score
                self._best_global_position = particle.best_position

    def get_swarm_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        positions: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((self._n_particles, 2), dtype=np.double)
        for index, particle in enumerate(self._particles):
            positions[index] = particle.position

        return positions

    @property
    def scene(self):  # type: ignore
        return self._scene

    @property
    def particles(self) -> list[ParticleBase]:
        return self._particles


class SwarmCentralized(SwarmBase):
    def __init__(self, n_particles: int, n_iterations: int, scene):  # type: ignore
        super().__init__(n_particles, n_iterations, scene)

        self._particles: list[ParticleCentralized] = []  # type: ignore
        for i in range(self._n_particles):
            self._particles.append(ParticleCentralized(self))

        self.update_global_information()

        plt.ion()

        if self._scene.verbosity.value > 1:
            self.show_current_position("Начальное положение")

    def get_information_from_particle(self, particle: ParticleCentralized) -> None:
        if particle.best_score > self._best_global_score:
            self._best_global_score = particle.best_score
            self._best_global_position = particle.best_position

    @property
    def best_global_score(self) -> float:
        return self._best_global_score

    @best_global_score.setter
    def best_global_score(self, new_value: float) -> None:
        self._best_global_score = new_value

    @property
    def best_global_position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._best_global_position

    @best_global_position.setter
    def best_global_position(self, new_position: np.ndarray[tp.Any, np.dtype[np.float64]]) -> None:
        self._best_global_position = new_position

    def show_current_position(self, title: str) -> None:
        backend = matplotlib.get_backend()
        if backend == "agg":
            return

        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = self.get_swarm_positions()

        figure = pickle.load(open("./stored_field/field.pickle", "rb"))
        ax = plt.gca()

        x, y = 100, 100

        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            figure.canvas.manager.window.move(x, y)

        ax.scatter(coordinates[:, 0] * self._scene.field.quality_scale,
                   coordinates[:, 1] * self._scene.field.quality_scale,
                   marker='o', color='b', ls='', s=40)

        ax.set_xlim(0, self.scene.field.width * self._scene.field.quality_scale)
        ax.set_ylim(0, self.scene.field.height * self._scene.field.quality_scale)
        ax.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            ax.annotate(label, (coordinates[index][0] * self._scene.field.quality_scale,
                                coordinates[index][1] * self._scene.field.quality_scale), fontsize=10)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.)
        plt.close(figure)

    def run(self) -> tuple[int | float, ...]:
        early_stopping_small_velocity_count = 0
        early_stopping_small_velocity = False

        early_stopping_around_answer = False

        eps_position = self._scene.hyperparameters.early_stopping["around_point"]["epsilon"]
        ratio_position = self._scene.hyperparameters.early_stopping["around_point"]["ratio"]

        eps_velocity = self._scene.hyperparameters.early_stopping["velocity"]["epsilon"]
        ratio_velocity = self._scene.hyperparameters.early_stopping["velocity"]["ratio"]

        for i in range(1, self._n_iterations + 1):
            for j in range(self._n_particles):
                self._particles[j].update()

            for j in range(ceil(ratio_position * self._n_particles)):
                early_stopping_around_answer_count = 0
                for k in range(self._n_particles):
                    if np.linalg.norm(self._particles[j].position - self._particles[k].position) < eps_position:
                        early_stopping_around_answer_count += 1

                if early_stopping_around_answer_count > ratio_position * self._n_particles:
                    early_stopping_around_answer = True
                    break

            for j in range(self._n_particles):
                if np.linalg.norm(self._particles[j].velocity) < eps_velocity:
                    early_stopping_small_velocity_count += 1

            if early_stopping_around_answer:
                break

            elif early_stopping_small_velocity_count > ratio_velocity * self._n_particles:
                early_stopping_small_velocity = True
                break

            if i % self._scene.verbosity.show_period == 0:
                if self._scene.verbosity.value > 0:
                    self.show_current_position(f"Итерация №{i}")

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

        if self._scene.verbosity.value > 0:
            self.show_current_position("Последняя итерация")

        return tuple(result)


class SwarmDecentralizedBase(SwarmBase):
    def __init__(self, n_particles: int, n_iterations: int, connection_radius: float, scene):  # type: ignore
        super().__init__(n_particles, n_iterations, scene)

        self.connection_radius = connection_radius

    def show_current_position(self, title: str) -> None:
        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = self.get_swarm_positions()

        figure = pickle.load(open("./stored_field/field.pickle", "rb"))
        ax = plt.gca()

        x, y = 100, 100

        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            figure.canvas.manager.window.move(x, y)

        ax.scatter(coordinates[:, 0]*self._scene.field.quality_scale,
                   coordinates[:, 1]*self._scene.field.quality_scale,
                   marker='o', color='b', ls='', s=20)

        ax.set_xlim(0, self.scene.field.width*self._scene.field.quality_scale)
        ax.set_ylim(0, self.scene.field.height*self._scene.field.quality_scale)
        ax.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            ax.annotate(label, (coordinates[index][0]*self._scene.field.quality_scale,
                                coordinates[index][1]*self._scene.field.quality_scale), fontsize=10)

        for coordinate in coordinates:
            circle = mpatches.Circle(coordinate*self._scene.field.quality_scale,
                                     self.connection_radius*self._scene.field.height*self._scene.field.quality_scale,
                                     color="g", fill=False, linestyle="--")
            ax.add_patch(circle)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.)
        plt.close(figure)

    def get_best_global_score(self, update: bool = True) -> float:
        if update:
            self.update_global_information()

        return self._best_global_score

    def get_best_global_position(self, update: bool = True) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        if update:
            self.update_global_information()

        return self._best_global_position


class SwarmDecentralized(SwarmDecentralizedBase):
    def __init__(self, n_particles: int, n_iterations: int, connection_radius: float, scene):  # type: ignore
        super().__init__(n_particles, n_iterations, connection_radius, scene)

        self._particles: list[ParticleDecentralized] = []  # type: ignore
        for i in range(self._n_particles):
            self._particles.append(ParticleDecentralized(self))

        self.update_global_information()

        plt.ion()

        if self._scene.verbosity.value > 1:
            self.show_current_position("Начальное положение")

    def run(self) -> tuple[int | float, ...]:
        early_stopping_small_velocity_count = 0
        early_stopping_small_velocity = False

        early_stopping_around_answer = False

        eps_position = self._scene.hyperparameters.early_stopping["around_point"]["epsilon"]
        ratio_position = self._scene.hyperparameters.early_stopping["around_point"]["ratio"]

        eps_velocity = self._scene.hyperparameters.early_stopping["velocity"]["epsilon"]
        ratio_velocity = self._scene.hyperparameters.early_stopping["velocity"]["ratio"]

        for i in range(1, self._n_iterations + 1):
            for j in range(self._n_particles):
                self._particles[j].update()

            for j in range(self._n_particles):
                self._particles[j].update_my_global_information()

            for j in range(ceil(ratio_position*self._n_particles)):
                early_stopping_around_answer_count = 0
                for k in range(self._n_particles):
                    if np.linalg.norm(self._particles[j].position - self._particles[k].position) < eps_position:
                        early_stopping_around_answer_count += 1

                if early_stopping_around_answer_count > ratio_position * self._n_particles:
                    early_stopping_around_answer = True
                    break

            for j in range(self._n_particles):
                if np.linalg.norm(self._particles[j].velocity) < eps_velocity:
                    early_stopping_small_velocity_count += 1

            if early_stopping_around_answer:
                break
            elif early_stopping_small_velocity_count > ratio_velocity * self._n_particles:
                early_stopping_small_velocity = True
                break

            if i % self._scene.verbosity.show_period == 0:
                if self._scene.verbosity.value > 0:
                    self.show_current_position(str(i))

            early_stopping_small_velocity_count = 0

            self._scene.inertia_scheduler.step()

        total_path_length: float = 0
        for j in range(self._n_particles):
            total_path_length += self._particles[j].path_length

        result = [i,
                  self._scene.answer.value - self.get_best_global_score(),
                  (self._scene.answer.value - self.get_best_global_score(False)) / self._scene.answer.value,
                  total_path_length,
                  total_path_length/self._n_particles]

        if early_stopping_around_answer:
            result = result + [1]
        elif early_stopping_small_velocity:
            result = result + [2]
        else:
            result = result + [0]

        if self._scene.verbosity.value > 0:
            self.show_current_position(f"Итерация №{i}")

        return tuple(result)


class SwarmCorrupted(SwarmDecentralizedBase):
    def __init__(self, n_particles: int, n_iterations: int, connection_radius: float, scene) -> None:  # type: ignore
        super().__init__(n_particles, n_iterations, connection_radius, scene)

        self.connection_radius = connection_radius

        self._particles: list[ParticleCorrupted] = []  # type: ignore
        for i in range(self._n_particles):
            self._particles.append(ParticleCorrupted(self))

        self.update_global_information()

        plt.ion()

        if self._scene.verbosity.value > 1:
            self.show_current_position("Начальное положение")

    def update_global_information(self) -> None:
        for particle in self._particles:
            if self._best_global_score < self.scene.field.target_function(*particle.best_position):
                self._best_global_score = self.scene.field.target_function(*particle.best_position)
                self._best_global_position = particle.best_position

    def run(self) -> tuple[int | float, ...]:
        early_stopping_small_velocity_count = 0
        early_stopping_small_velocity = False

        early_stopping_around_answer = False

        eps_position = self._scene.hyperparameters.early_stopping["around_point"]["epsilon"]
        ratio_position = self._scene.hyperparameters.early_stopping["around_point"]["ratio"]

        eps_velocity = self._scene.hyperparameters.early_stopping["velocity"]["epsilon"]
        ratio_velocity = self._scene.hyperparameters.early_stopping["velocity"]["ratio"]

        for i in range(1, self._n_iterations + 1):
            for j in range(self._n_particles):
                self._particles[j].update()

            for j in range(self._n_particles):
                self._particles[j].update_my_global_information()

            for j in range(ceil(ratio_position*self._n_particles)):
                early_stopping_around_answer_count = 0
                for k in range(self._n_particles):
                    if np.linalg.norm(self._particles[j].position - self._particles[k].position) < eps_position:
                        early_stopping_around_answer_count += 1

                if early_stopping_around_answer_count > ratio_position * self._n_particles:
                    early_stopping_around_answer = True
                    break

            for j in range(self._n_particles):
                if np.linalg.norm(self._particles[j].velocity) < eps_velocity:
                    early_stopping_small_velocity_count += 1

            if early_stopping_around_answer:
                break

            elif early_stopping_small_velocity_count > ratio_velocity * self._n_particles:
                early_stopping_small_velocity = True
                break

            if i % self._scene.verbosity.show_period == 0:
                if self._scene.verbosity.value > 1:
                    self.show_current_position(f"Итерация №{i}")

            early_stopping_small_velocity_count = 0

            self._scene.inertia_scheduler.step()

        total_path_length: float = 0
        for j in range(self._n_particles):
            total_path_length += self._particles[j].path_length

        result = [i,
                  self._scene.answer.value - self.get_best_global_score(),
                  (self._scene.answer.value - self.get_best_global_score(False)) / self._scene.answer.value,
                  total_path_length,
                  total_path_length/self._n_particles]

        if early_stopping_around_answer:
            result = result + [1]
        elif early_stopping_small_velocity:
            result = result + [2]
        else:
            result = result + [0]

        if self._scene.verbosity.value > 0:
            self.show_current_position("Последняя итерация")

        return tuple(result)
