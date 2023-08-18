import typing as tp
import json
from dataclasses import dataclass

import numpy as np
from numpy.random import uniform, normal

from field import field as fl
from logger import custom_logger
from swarm import swarm as sw

from gradient_methods import gradient_methods

# cause making a json from function isn't trivial, I've used this mappings from str to function object

function_name2object = {"gaussian": fl.gaussian}
symbolic_function_name2object = {"gaussian": fl.gaussian_symbolic}


class InertiaScheduler:
    def __init__(self, step_size: int, gamma: float, scene: 'Scene'):
        self._step_size: int = step_size
        self._gamma: float = gamma
        self._steps: int = 0
        self._scene = scene

    def step(self):
        self._steps += 1
        if self._steps % self._step_size == 0:
            self._scene.hyperparameters.w = self._scene.hyperparameters.w * self._gamma


@dataclass()
class Verbosity:
    _value: float
    _show_period: int

    @property
    def value(self):
        return self._value

    @property
    def show_period(self):
        return self._show_period


@dataclass()
class HyperparametersCentralizedSwarm:
    _w: float
    _c1: float
    _c2: float

    _velocity_factor: float

    _early_stopping: dict[str, dict[str, float]]

    _verbosity: Verbosity

    _position_factor: float

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, new_value: float):
        self._w = new_value

    @property
    def c1(self):
        return self._c1

    @property
    def c2(self):
        return self._c2

    @property
    def velocity_factor(self):
        return self._velocity_factor

    @property
    def early_stopping(self):
        return self._early_stopping

    @property
    def verbosity(self):
        return self._verbosity

    @property
    def position_factor(self):
        return self._position_factor


@dataclass()
class HyperparametersDecentralizedSwarm(HyperparametersCentralizedSwarm):
    _connection_radius: float

    @property
    def connection_radius(self):
        return self._connection_radius


@dataclass()
class HyperparametersCorruptedSwarm(HyperparametersDecentralizedSwarm):
    _noise: dict[str, str | float]

    @property
    def noise(self):
        return self._noise


class Answer:
    def __init__(self, position: np.ndarray, target_function: str):
        self._position: np.ndarray = np.array(position)
        self._value: float = function_name2object[target_function](*position)

    @property
    def position(self):
        return self._position

    @property
    def value(self):
        return self._value


@dataclass()
class Noise:
    _noise_type: str
    _noise_scale: float

    def add_noise(self, particle_position: np.ndarray, answer_position: np.ndarray):
        if self._noise_type == "normal":
            return normal(0, np.linalg.norm(particle_position - answer_position)) * self._noise_scale
        elif self._noise_type == "uniform":
            return uniform(-np.linalg.norm(particle_position - answer_position),
                           np.linalg.norm(particle_position - answer_position)) * self._noise_scale

        raise ValueError("Please, check the 'noise type' field at your config;")


@dataclass
class _Hyperparameters:
    _w: float
    _c1: float
    _c2: float
    _velocity_scale: float
    _inertia_scheduler_step_size: int
    _inertia_scheduler_gamma: float
    _connect_radius: float
    _noise_scale: float
    _position_factor: float

    @property
    def noise_scale(self):
        return self._noise_scale

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
    def velocity_scale(self) -> float:
        return self._velocity_scale

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

    @property
    def position_factor(self):
        return self._position_factor


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
                                 target_function=function_name2object[config["field"]["target_function"]],
                                 target_function_symbolic=
                                 symbolic_function_name2object[config["field"]["target_function"]])
        else:
            raise ValueError("Please, check your 'target_function'; It must be present in the 'function_name2object'")

        if config["solver"]["type"] == "swarm":
            self.inertia_scheduler: InertiaScheduler = \
                InertiaScheduler(step_size=config["solver"]["hyperparams"]["inertia_scheduler"]["step_size"],
                                 gamma=config["solver"]["hyperparams"]["inertia_scheduler"]["gamma"],
                                 scene=self)

            self.verbosity: Verbosity = Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                  _show_period=config["solver"]["verbosity"]["show_period"])

            if config["solver"]["specification"] == "centralized":
                self.solver: sw.SwarmCentralized = \
                    sw.SwarmCentralized(n_particles=config["solver"]["hyperparams"]["n_particles"],
                                        n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                                        scene=self)
                self.hyperparameters: HyperparametersCentralizedSwarm = \
                    HyperparametersCentralizedSwarm(_w=config["solver"]["hyperparams"]["coefficients"]["w"],
                                                    _c1=config["solver"]["hyperparams"]["coefficients"]["c1"],
                                                    _c2=config["solver"]["hyperparams"]["coefficients"]["c2"],
                                                    _velocity_factor=config["solver"]["hyperparams"]["velocity_factor"],
                                                    _verbosity=Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                    _show_period=config["solver"]["verbosity"]["show_period"]),
                                                    _early_stopping=config["solver"]["hyperparams"]["early_stopping"],
                                                    _position_factor=config["solver"]["hyperparams"]["position_factor"])
            elif config["solver"]["specification"] == "decentralized":
                self.solver: sw.SwarmDecentralized =\
                    sw.SwarmDecentralized(n_particles=config["solver"]["hyperparams"]["n_particles"],
                                          n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                                          connection_radius=config["solver"]["hyperparams"]["connection_radius"],
                                          scene=self)
                self.hyperparameters: HyperparametersDecentralizedSwarm = \
                    HyperparametersDecentralizedSwarm(_w=config["solver"]["hyperparams"]["coefficients"]["w"],
                                                      _c1=config["solver"]["hyperparams"]["coefficients"]["c1"],
                                                      _c2=config["solver"]["hyperparams"]["coefficients"]["c2"],
                                                      _velocity_factor=
                                                      config["solver"]["hyperparams"]["velocity_factor"],
                                                      _verbosity=
                                                      Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                      _show_period=config["solver"]["verbosity"]["show_period"]),
                                                      _early_stopping=config["solver"]["hyperparams"]["early_stopping"],
                                                      _connection_radius=
                                                      config["solver"]["hyperparams"]["connection_radius"],
                                                      _position_factor=
                                                      config["solver"]["hyperparams"]["position_factor"])
            elif config["solver"]["specification"] == "corrupted":
                self.solver: sw.SwarmCorrupted = \
                    sw.SwarmCorrupted(n_particles=config["solver"]["hyperparams"]["n_particles"],
                                      n_iterations=config["solver"]["hyperparams"]["n_iterations"],
                                      connection_radius=config["solver"]["hyperparams"]["connection_radius"],
                                      scene=self)
                self.hyperparameters: HyperparametersCorruptedSwarm = \
                    HyperparametersCorruptedSwarm(_w=config["solver"]["hyperparams"]["coefficients"]["w"],
                                                  _c1=config["solver"]["hyperparams"]["coefficients"]["c1"],
                                                  _c2=config["solver"]["hyperparams"]["coefficients"]["c2"],
                                                  _velocity_factor=config["solver"]["hyperparams"]["velocity_factor"],
                                                  _verbosity=Verbosity(_value=config["solver"]["verbosity"]["value"],
                                                  _show_period=config["solver"]["verbosity"]["show_period"]),
                                                  _early_stopping=config["solver"]["hyperparams"]["early_stopping"],
                                                  _connection_radius=
                                                  config["solver"]["hyperparams"]["connection_radius"],
                                                  _noise=config["solver"]["noise"],
                                                  _position_factor=config["solver"]["hyperparams"]["position_factor"])
                self.noise: Noise = Noise(_noise_type=config["solver"]["noise"]["type"],
                                          _noise_scale=config["solver"]["noise"]["scale"])
            else:
                raise ValueError("Please, check your swarm solver's specification;")

            self.spawn_type: str = config["solver"]["spawn_type"]
            if self.spawn_type == "small_area":
                self.edge: int = np.random.randint(4)
                position_factor: float = self.hyperparameters.position_factor
                if self.edge == 0:  # left
                    self.spawn_start_location: np.ndarray = np.array(
                        [0, uniform(self.field.height / position_factor,
                                    self.field.height -
                                    self.field.height / position_factor)])
                elif self.edge == 1:  # right
                    self.spawn_start_location: np.ndarray = np.array([self.field.width,
                                                                      uniform(self.field.height / position_factor,
                                                                              self.field.height -
                                                                              self.field.height / position_factor)])
                elif self.edge == 2:  # top
                    self.spawn_start_location: np.ndarray = np.array([uniform(self.field.width / position_factor,
                                                                              self.field.width -
                                                                              self.field.width / position_factor),
                                                                      0])
                elif self.edge == 3:  # bottom
                    self.spawn_start_location: np.ndarray = np.array([uniform(self.field.width / position_factor,
                                                                              self.field.width -
                                                                              self.field.width / position_factor),
                                                                      self.field.height])

    def run(self) -> tuple[int|float, ...]:
        results = self.solver.run()

        return results


class _Scene:
    def __init__(self,
                 # field_height: float, field_width: float,
                 # field_target_function: tp.Callable[[float, float, tuple[float, float]], float],
                 swarm_n_particles: int, swarm_n_iterations: int,
                 spawn_type: str,
                 # answer_value: float, answer_location: np.ndarray,
                 verbose: int,
                 answer: Answer,
                 field: fl.GaussianField,
                 hyperparameters: Hyperparameters,
                 swarm_type: str = "centralized"):
        self.answer: Answer = answer
        self.field: fl.FieldInterface = field
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
        if swarm_type == "centralized":
            self.swarm: sw.SwarmBase = sw.SwarmCentralized(swarm_n_particles, swarm_n_iterations, self)
        elif swarm_type == "decentralized":
            self.swarm: sw.SwarmBase = sw.SwarmDecentralized(swarm_n_particles, swarm_n_iterations,
                                                             self.hyperparameters._connection_radius, self)
        elif swarm_type == "corrupted":
            self.swarm: sw.SwarmCorrupted = sw.SwarmCorrupted(swarm_n_particles, swarm_n_iterations,
                                                              self.hyperparameters._connect_radius, self)

class SceneGrad:
    def __init__(self, n_iterations, answer: Answer,
                 field: fl.GaussianField, verbose):
        self.answer = answer
        self.field = field
        self.verbose = verbose

        self.solver = gradient_methods.GradientLift(self, n_iterations)

    def run(self):
        results = self.solver.run()
        return results


if __name__ == "__main__":
    json_config = """
        {
        "field":
            {
            "target_function": "gaussian",
            "height": 10,
            "width": 10,
            "correctness_scale": 100
             },
        "answer": [5, 5],
        "solver":
            {
            "type": "decentralized",
            "spawn_type": "small_area",
            "n_iterations": 1000,
            "n_particles": 10,
            "connection_radius": 0.1,
            "velocity_factor": 50,
            "verbosity":
                {
                "value": 2,
                "show_period": 10
                },
            "noise":
                {
                "type": "normal",
                "scale": 0.00126
                },
            "hyperparams":
                {
                "coefficients":
                    {
                    "w": 1,
                    "c1": 2,
                    "c2": 2
                    },
                "inertia_scheduler":
                    {
                    "gamma": 0.75,
                    "step_size": 100
                    },    
                "early_stopping":
                    {
                    "around_point": 
                        {
                        "epsilon": 0.0001,
                        "ratio": 0.75
                        },
                    "velocity":
                        {
                        "epsilon": 0.0001,
                        "ratio": 0.75
                        }
                    }
                }
            }
        }
    """

    height = 10.
    width = 10.

    answer = Answer(fl.gaussian(height/2, width/2, (height/2, width/2)), np.array([height/2, width/2]))
    field = fl.GaussianField(height, width, fl.gaussian, fl.gaussian_symbolic)
    hyperparams = Hyperparameters(_w=1, _c1=2, _c2=2, _velocity_scale=50,
                                  _inertia_scheduler_step_size=100, _inertia_scheduler_gamma=0.75,
                                  _connect_radius=height*0.1)

    my_scene = Scene(20, 1000, "small_area", 2, answer, field, hyperparams, "centralized")
    _ = my_scene.run()
    print(_)

    """
    height = 10.
    width = 10.

    answer = Answer(fl.gaussian(height / 2, width / 2), np.array([height / 2, width / 2]))
    field = fl.Field(height, width, fl.gaussian)

    my_logger = custom_logger.CustomLogger("./logs/gradient/gradient_lift.csv")
    # columns = ["n_iterations", "absolute_error", "relative_error", "total_path"]
    # my_logger.write(columns, "w")

    for i in range(1000-943):
        my_scene = SceneGrad(3000, answer, field, 0)
        results = list(my_scene.run())
        results = results
        my_logger.write(results, "a")
    """
