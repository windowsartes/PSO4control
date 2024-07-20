import os
import typing as tp
import pickle
from abc import ABC, abstractmethod
from math import ceil

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# from src.solvers.solver_interface import SolverInterface
from src.solvers.swarm.particle import Particle


class SwarmInterface:
    @abstractmethod
    def update_scores(
        self,
        particles_scores: list[float],
    ) -> None:
        pass

    @abstractmethod
    def correct_positions(self) -> None:
        pass

    @abstractmethod
    def turn(self) -> None:
        pass

    @abstractmethod
    def get_swarm_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def show(
        self,
        title: str,
    ):
        pass


class SwarmBase(SwarmInterface):
    def get_swarm_positions(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        positions: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((len(self._particles), 2), dtype=np.double)
            
        for index, particle in enumerate(self._particles):
            positions[index] = particle.position

        return positions

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    def correct_positions(
        self,
        size: float,
    ) -> None:
        for i in range(len(self._particles)):  
            self._particles[i].position[0] = max(self._particles[i].position[0], 0)
            self._particles[i].position[0] = min(self._particles[i].position[0], size)

            self._particles[i].position[1] = max(self._particles[i].position[1], 0)
            self._particles[i].position[1] = min(self._particles[i].position[1], size)


class SwarmCentralized(SwarmBase):
    def __init__(
        self,
        n_particles: int,
        field_size,
        spawn_type: str,
        position_factor: float,
        velocity_factor: float,
        w: float,
        c1: float,
        c2: float,
        spawn_start_location: np.ndarray[tp.Any, np.dtype[np.float64]] = np.ones((2)),
        spawn_edge: int = 1,
    ):
        self._particles: list[Particle] = []
        for i in range(n_particles):
            self._particles.append(
                Particle(
                    field_size,
                    spawn_type,
                    position_factor,
                    velocity_factor,
                    w,
                    c1,
                    c2,
                    spawn_start_location,
                    spawn_edge,
                )
            )

        self._best_global_score: float = 0.
        self._best_global_position: np.ndarray[tp.Any, np.dtype[np.float64]] = np.empty((2))

    def update_scores(
        self,
        particles_scores: list[float],
    ) -> None:
        for i in range(len(self._particles)):
            if particles_scores[i] > self._particles[i].best_score:
                self._particles[i].best_score = particles_scores[i]
                self._particles[i]._best_position = self._particles[i].position

            if particles_scores[i] > self._best_global_score:
                self._best_global_score = particles_scores[i]
                self._best_global_position = self._particles[i].position

    def turn(self):
        for i in range(len(self._particles)):
            self._particles[i].move(self._best_global_position)

    def show(self, title: str):
        backend = matplotlib.get_backend()
    
        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = self.get_swarm_positions()
        
        with open("./stored_field/field.pickle", "rb") as f:
            figure = pickle.load(f)
        ax = plt.gca()

        x, y = 100, 100

        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))

        quality_scale = 100

        ax.scatter(
            coordinates[:, 0] * quality_scale,
            coordinates[:, 1] * quality_scale,
            marker='o',
            color='b',
            ls='',
            s=40,
        )

        ax.set_xlim(0, 10 * quality_scale)
        ax.set_ylim(0, 10 * quality_scale)
        ax.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            ax.annotate(
                label,
                (
                    coordinates[index][0] * quality_scale,
                    coordinates[index][1] * quality_scale,
                ),
                fontsize=10,
            )

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.)
        plt.close(figure)


class SwarmDecentralized(SwarmBase):
    def __init__(
        self,
        n_particles: int,
        field_size: float,
        spawn_type: str,
        position_factor: float,
        velocity_factor: float,
        w: float,
        c1: float,
        c2: float,
        spawn_start_location: np.ndarray[tp.Any, np.dtype[np.float64]] = np.ones((2)),
        spawn_edge: int = 1,
        connection_radius: float = 0.5,
    ):
        self._particles: list[Particle] = []
        for i in range(n_particles):
            self._particles.append(
                Particle(
                    field_size,
                    spawn_type,
                    position_factor,
                    velocity_factor,
                    w,
                    c1,
                    c2,
                    spawn_start_location,
                    spawn_edge,
                )
            )

        self._connection_radius = connection_radius

        self._best_global_scores: list[float] = [0.0] * n_particles
        self._best_global_positions: list[np.ndarray[tp.Any, np.dtype[np.float64]]] = [np.empty((2))] * n_particles

    def update_scores(
        self,
        particles_scores: list[float],
    ) -> None:
        field_size: float = 10.0
        for i in range(len(self._particles)):
            if particles_scores[i] > self._particles[i].best_score:
                self._particles[i].best_score = particles_scores[i]
                self._particles[i]._best_position = self._particles[i].position

        for i in range(len(self._particles)):
            for j in range(len(self._particles)):
                if np.linalg.norm(self._particles[i].position - self._particles[j].position) < \
                    self._connection_radius * field_size:
                    if self._best_global_scores[i] < self._particles[j].best_score:
                        self._best_global_scores[i] = self._particles[j].best_score
                        self._best_global_positions[i] = self._particles[j].best_position

    def turn(self):
        for i in range(len(self._particles)):
            self._particles[i].move(self._best_global_positions[i])

    def show(self, title: str):
        backend = matplotlib.get_backend()
    
        coordinates: np.ndarray[tp.Any, np.dtype[np.float64]] = self.get_swarm_positions()
        
        with open("./stored_field/field.pickle", "rb") as f:
            figure = pickle.load(f)
        ax = plt.gca()

        x, y = 100, 100

        if backend == 'TkAgg':
            figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            figure.canvas.manager.window.SetPosition((x, y))

        quality_scale = 100
        field_size: float = 10.0

        ax.scatter(
            coordinates[:, 0] * quality_scale,
            coordinates[:, 1] * quality_scale,
            marker='o',
            color='b',
            ls='',
            s=40,
        )

        ax.set_xlim(0, 10 * quality_scale)
        ax.set_ylim(0, 10 * quality_scale)
        ax.set_title(title)

        for index, label in enumerate(np.arange(len(coordinates))):
            ax.annotate(
                label,
                (
                    coordinates[index][0] * quality_scale,
                    coordinates[index][1] * quality_scale,
                ),
                fontsize=10,
            )

        for coordinate in coordinates:
            circle = mpatches.Circle(
                coordinate * quality_scale,
                self._connection_radius * field_size * quality_scale,
                color="g",
                fill=False,
                linestyle="--",
            )
            ax.add_patch(circle)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.)
        plt.close(figure)
