import json
import typing as tp
from abc import ABC
from dataclasses import dataclass

import click
import numpy as np
from numpy.random import uniform, normal

import src.solvers.solver_interface
from src.field import field as fl
from src.solvers.swarm import swarm as sw
from src.solvers.gradient_methods import gradient_methods


# cause making a json from function isn't trivial, I've used these mappings from str to function object
function_name2object = {"gaussian": fl.gaussian}
symbolic_function_name2object = {"gaussian": fl.gaussian_symbolic}


class InertiaScheduler:
    def __init__(self, step_size: int, gamma: float, scene: 'Scene'):
        self._step_size: int = step_size
        self._gamma: float = gamma
        self._steps: int = 0
        self._scene: Scene = scene

    def step(self) -> None:
        self._steps += 1
        if self._steps % self._step_size == 0:
            if isinstance(self._scene.hyperparameters, HyperparametersCentralizedSwarm):
                self._scene.hyperparameters.w = self._scene.hyperparameters.w * self._gamma
            else:
                raise ValueError("There is an error during initialization: your 'hyperparameters' field"
                                 "must be 'HyperparametersCentralizedSwarm' instance or its heir")


@dataclass()
class Verbosity:
    _value: float
    _show_period: int

    @property
    def value(self) -> float:
        return self._value

    @property
    def show_period(self) -> float:
        return self._show_period


@dataclass()
class HyperparametersInterface(ABC):
    pass


@dataclass()
class HyperparametersCentralizedSwarm(HyperparametersInterface):
    _w: float
    _c1: float
    _c2: float

    _velocity_factor: float

    _early_stopping: dict[str, dict[str, float]]

    _verbosity: Verbosity

    _position_factor: float

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, new_value: float) -> None:
        self._w = new_value

    @property
    def c1(self) -> float:
        return self._c1

    @property
    def c2(self) -> float:
        return self._c2

    @property
    def velocity_factor(self) -> float:
        return self._velocity_factor

    @property
    def early_stopping(self) -> dict[str, dict[str, float]]:
        return self._early_stopping

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    @property
    def position_factor(self) -> float:
        return self._position_factor


@dataclass()
class HyperparametersDecentralizedSwarm(HyperparametersCentralizedSwarm):
    _connection_radius: float

    @property
    def connection_radius(self) -> float:
        return self._connection_radius


@dataclass()
class HyperparametersCorruptedSwarm(HyperparametersDecentralizedSwarm):
    _noise: dict[str, str | float]

    @property
    def noise(self) -> dict[str, str | float]:
        return self._noise


@dataclass()
class HyperparametersGradientMethod(HyperparametersInterface):
    _velocity_factor: float
    _verbosity: Verbosity
    _early_stopping_epsilon: float

    @property
    def velocity_factor(self) -> float:
        return self._velocity_factor

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    @property
    def early_stopping_epsilon(self) -> float:
        return self._early_stopping_epsilon


class Answer:
    def __init__(self, position: np.ndarray[tp.Any, np.dtype[np.float64]], target_function: str) -> None:
        self._position: np.ndarray[tp.Any, np.dtype[np.float64]] = np.array(position)
        self._value: float = function_name2object[target_function](*position)

    @property
    def position(self) -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        return self._position

    @property
    def value(self) -> float:
        return self._value


@dataclass()
class Noise:
    _noise_type: str
    _noise_scale: float

    def add_noise(self, particle_position: np.ndarray[tp.Any, np.dtype[np.float64]],
                  answer_position: np.ndarray[tp.Any, np.dtype[np.float64]])\
            -> np.ndarray[tp.Any, np.dtype[np.float64]]:
        if self._noise_type == "normal":
            return normal(0, np.linalg.norm(particle_position - answer_position)) * self._noise_scale
        elif self._noise_type == "uniform":
            return uniform(-np.linalg.norm(particle_position - answer_position),
                           np.linalg.norm(particle_position - answer_position)) * self._noise_scale

        raise ValueError("Please, check the 'noise type' field at your config;")


class Scene:
    def __init__(self, path_to_config: str):
        with open(path_to_config, "r") as config_file:
            config = json.load(config_file)
            self.config = config

        self.answer: Answer = Answer(position=config["answer"],
                                     target_function=config["field"]["target_function"])

        if config["field"]["target_function"] == "gaussian":
            self.field: fl.FieldInterface = \
                fl.GaussianField(height=config["field"]["height"],
                                 width=config["field"]["width"],
                                 quality_scale=config["field"]["quality_scale"],
                                 target_function=function_name2object[config["field"]["target_function"]],
                                 target_function_symbolic=symbolic_function_name2object[config["field"]
                                 ["target_function"]])
        else:
            raise ValueError("Please, check your 'target_function'; It must be present in the 'function_name2object'")

        if config["solver"]["type"] == "swarm":
            self.inertia_scheduler: InertiaScheduler = \
                InertiaScheduler(step_size=config["solver"]["hyperparams"]["inertia_scheduler"]["step_size"],
                                 gamma=config["solver"]["hyperparams"]["inertia_scheduler"]["gamma"],
                                 scene=self)

            self.verbosity: Verbosity = Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                  _show_period=config["solver"]["verbosity"]["show_period"])

            self.spawn_type: str = config["solver"]["spawn_type"]
            if self.spawn_type == "small_area":
                self.edge: int = np.random.randint(4)
                position_factor: float = config["solver"]["hyperparams"]["position_factor"]
                self.spawn_start_location: np.ndarray[tp.Any, np.dtype[np.float64]]
                if self.edge == 0:  # left
                    self.spawn_start_location = np.array([0, uniform(self.field.height / position_factor,
                                                          self.field.height - self.field.height / position_factor)])
                elif self.edge == 1:  # right
                    self.spawn_start_location = np.array([self.field.width, uniform(self.field.height / position_factor,
                                                          self.field.height - self.field.height / position_factor)])
                elif self.edge == 2:  # top
                    self.spawn_start_location = np.array([uniform(self.field.width / position_factor,
                                                          self.field.width - self.field.width / position_factor), 0])
                elif self.edge == 3:  # bottom
                    self.spawn_start_location = np.array([uniform(self.field.width / position_factor,
                                                          self.field.width - self.field.width / position_factor),
                                                          self.field.height])
            self.hyperparameters: HyperparametersInterface
            self.solver: solvers.solver_interface.SolverInterface
            if config["solver"]["specification"] == "centralized":
                self.hyperparameters = \
                    HyperparametersCentralizedSwarm(_w=config["solver"]["hyperparams"]["coefficients"]["w"],
                                                    _c1=config["solver"]["hyperparams"]["coefficients"]["c1"],
                                                    _c2=config["solver"]["hyperparams"]["coefficients"]["c2"],
                                                    _velocity_factor=config["solver"]["hyperparams"]["velocity_factor"],
                                                    _verbosity=Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                                         _show_period=config["solver"]["verbosity"][
                                                                             "show_period"]),
                                                    _early_stopping=config["solver"]["hyperparams"]["early_stopping"],
                                                    _position_factor=config["solver"]["hyperparams"]["position_factor"])
                self.solver = \
                    sw.SwarmCentralized(n_particles=config["solver"]["hyperparams"]["n_particles"],
                                        n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                                        scene=self)
            elif config["solver"]["specification"] == "decentralized":
                self.hyperparameters = \
                    HyperparametersDecentralizedSwarm(_w=config["solver"]["hyperparams"]["coefficients"]["w"],
                                                      _c1=config["solver"]["hyperparams"]["coefficients"]["c1"],
                                                      _c2=config["solver"]["hyperparams"]["coefficients"]["c2"],
                                                      _velocity_factor=config["solver"]["hyperparams"]
                                                      ["velocity_factor"],
                                                      _verbosity=Verbosity(_value=config["solver"]["verbosity"]
                                                      ["value"], _show_period=config["solver"]["verbosity"][
                                                                    "show_period"]),
                                                      _early_stopping=config["solver"]["hyperparams"]["early_stopping"],
                                                      _connection_radius=config["solver"]["hyperparams"]
                                                      ["connection_radius"],
                                                      _position_factor=config["solver"]["hyperparams"]
                                                      ["position_factor"])
                self.solver =\
                    sw.SwarmDecentralized(n_particles=config["solver"]["hyperparams"]["n_particles"],
                                          n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                                          connection_radius=config["solver"]["hyperparams"]["connection_radius"],
                                          scene=self)
            elif config["solver"]["specification"] == "corrupted":
                self.hyperparameters = \
                    HyperparametersCorruptedSwarm(_w=config["solver"]["hyperparams"]["coefficients"]["w"],
                                                  _c1=config["solver"]["hyperparams"]["coefficients"]["c1"],
                                                  _c2=config["solver"]["hyperparams"]["coefficients"]["c2"],
                                                  _velocity_factor=config["solver"]["hyperparams"]["velocity_factor"],
                                                  _verbosity=Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                                       _show_period=config["solver"]["verbosity"][
                                                                           "show_period"]),
                                                  _early_stopping=config["solver"]["hyperparams"]["early_stopping"],
                                                  _connection_radius=config["solver"]["hyperparams"]
                                                  ["connection_radius"],
                                                  _noise=config["solver"]["noise"],
                                                  _position_factor=config["solver"]["hyperparams"]["position_factor"])
                self.noise: Noise = Noise(_noise_type=config["solver"]["noise"]["type"],
                                          _noise_scale=config["solver"]["noise"]["scale"])
                self.solver = \
                    sw.SwarmCorrupted(n_particles=config["solver"]["hyperparams"]["n_particles"],
                                      n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                                      connection_radius=config["solver"]["hyperparams"]["connection_radius"],
                                      scene=self)
            else:
                raise ValueError("Please, check your swarm solver's specification;")
        elif config["solver"]["type"] == "gradient":
            self.verbosity = Verbosity(_value=config["solver"]["verbosity"]["value"],
                                       _show_period=config["solver"]["verbosity"]["show_period"])
            if config["solver"]["specification"] == "lift":
                self.hyperparameters = \
                    HyperparametersGradientMethod(_verbosity=Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                                       _show_period=config["solver"]["verbosity"][
                                                                           "show_period"]),
                                                  _velocity_factor=config["solver"]["hyperparams"]["velocity_factor"],
                                                  _early_stopping_epsilon=config["solver"]["hyperparams"]
                                                  ["early_stopping_epsilon"])
                self.solver = gradient_methods.GradientLift(
                    n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                    scene=self)
            elif config["solver"]["specification"] == "newton":
                self.hyperparameters = \
                    HyperparametersGradientMethod(_verbosity=Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                                       _show_period=config["solver"]["verbosity"][
                                                                           "show_period"]),
                                                  _velocity_factor=config["solver"]["hyperparams"]["velocity_factor"],
                                                  _early_stopping_epsilon=config["solver"]["hyperparams"]
                                                  ["early_stopping_epsilon"])
                self.solver = gradient_methods.NewtonMethod(
                    n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                    scene=self)
            else:
                raise ValueError("Please, check your solver's specification - it must be either 'lift' or 'newton'")
        else:
            raise ValueError("Please, check your solver's type: it must be either 'swarm' or 'gradient';")

    def run(self) -> tuple[int | float, ...]:
        result = self.solver.run()

        return result


@click.command()
@click.argument("config", type=click.Path(exists=True))
def cli(config: click.Path(exists=True)) -> None:  # type: ignore
    my_scene: Scene = Scene(config)
    result = my_scene.run()

    print(result)


if __name__ == "__main__":
    cli()
