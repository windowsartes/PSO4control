import numpy as np
from numpy.random import uniform

import pickle

import matplotlib
from matplotlib import pyplot as plt


class GradientLift:
    def __init__(self, scene, n_iterations: int):
        self._scene = scene
        self._score = float("-inf")
        self._n_iterations = n_iterations

        self._total_path_length = 0

        edge: int = np.random.randint(4)
        if edge == 0:  # left
            self._position: np.ndarray = np.array([0, uniform(0, self._scene.field.height)])
        elif edge == 1:  # right
            self._position: np.ndarray = np.array([self._scene.field.width, uniform(0, self._scene.field.height)])
        elif edge == 2:  # top
            self._position: np.ndarray = np.array([uniform(0, self._scene.field.width), 0])
        else:  # bottom
            self._position: np.ndarray = np.array([uniform(0, self._scene.field.width), self._scene.field.height])

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_value: np.ndarray):
        self._position = new_value
        self._correct_position()

    def run(self):
        eps_gradient: float = 0.00001
        for i in range(1, self._n_iterations + 1):
            current_gradient: np.ndarray = self._scene.field.gradient(*self._position)

            current_norm = np.linalg.norm(current_gradient)
            if current_norm < eps_gradient:
                break

            if current_norm >= (self._scene.field.height / 50):
                print("normed")
                current_gradient = (current_gradient / current_norm) * (self._scene.field.height / 50)

            self._position = self._position + current_gradient
            self._correct_position()
            self._total_path_length += np.linalg.norm(current_gradient)
            self._score = self._scene.field.target_function(*self._position)

            if self._scene.verbose > 0:
                if i % 500 == 0:
                    self.show_current_position(str(i))

        return [i,
                self._scene.answer.value - self._score,
                (self._scene.answer.value - self._score)/self._scene.answer.value,
                self._total_path_length]

    def _correct_position(self):
        if self._position[0] < 0:
            self._position[0] = 0
        elif self._position[0] > self._scene.field.width:
            self._position[0] = self._scene.field.width

        if self._position[1] < 0:
            self._position[1] = 0
        elif self._position[1] > self._scene.field.height:
            self._position[1] = self._scene.field.height

    def show_current_position(self, title: str):
        correctness_scale = 100

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

        ax.scatter(self.position[0] * correctness_scale, self.position[1] * correctness_scale,
                   marker='o', color='b', ls='', s=20)

        ax.set_xlim(0, self._scene.field.width * correctness_scale)
        ax.set_ylim(0, self._scene.field.height * correctness_scale)
        ax.set_title(title)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.5)
        plt.close(figure)


class NewtonMethod:
    def __init__(self, scene, n_iterations: int):
        self._scene = scene
        self._score = float("-inf")
        self._n_iterations = n_iterations

        self._total_path_length = 0

        edge: int = np.random.randint(4)
        if edge == 0:  # left
            self._position: np.ndarray = np.array([0, uniform(0, self._scene.field.height)])
        elif edge == 1:  # right
            self._position: np.ndarray = np.array([self._scene.field.width, uniform(0, self._scene.field.height)])
        elif edge == 2:  # top
            self._position: np.ndarray = np.array([uniform(0, self._scene.field.width), 0])
        else:  # bottom
            self._position: np.ndarray = np.array([uniform(0, self._scene.field.width), self._scene.field.height])

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_value: np.ndarray):
        self._position = new_value
        self._correct_position()

    def run(self):
        eps_gradient: float = 0.0001
        for i in range(1, self._n_iterations + 1):
            current_gradient: np.ndarray = self._scene.field.gradient(*self._position)
            hessian_inv = np.linalg.inv(self._scene.field.hessian(*self._position))

            current_shift = hessian_inv @ current_gradient

            current_norm = np.linalg.norm(current_shift)
            if current_norm < eps_gradient:
                break

            if current_norm >= (self._scene.field.height/100):
                current_shift = (current_shift/current_norm)*(self._scene.field.height/100)

            self._position = self._position - current_shift
            self._correct_position()

            self._total_path_length += np.linalg.norm(current_gradient)
            self._score = self._scene.field.target_function(*self._position)

            if self._scene.verbose > 0:
                if i % 500 == 0:
                    self.show_current_position(str(i))

        return [i,
                self._scene.answer.value - self._score,
                (self._scene.answer.value - self._score)/self._scene.answer.value,
                self._total_path_length]

    def _correct_position(self):
        if self._position[0] < 0:
            self._position[0] = 0
        elif self._position[0] > self._scene.field.width:
            self._position[0] = self._scene.field.width

        if self._position[1] < 0:
            self._position[1] = 0
        elif self._position[1] > self._scene.field.height:
            self._position[1] = self._scene.field.height

    def show_current_position(self, title: str):
        correctness_scale = 100

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

        ax.scatter(self.position[0] * correctness_scale, self.position[1] * correctness_scale,
                   marker='o', color='b', ls='', s=20)

        ax.set_xlim(0, self._scene.field.width * correctness_scale)
        ax.set_ylim(0, self._scene.field.height * correctness_scale)
        ax.set_title(title)

        plt.draw()
        plt.gcf().canvas.flush_events()

        plt.pause(2.5)
        plt.close(figure)
