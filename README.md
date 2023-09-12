# Particle Swarm Optimization for control theory

- [Particle Swarm Optimization for control theory](#particle-swarm-optimization-for-control-theory)
  - [Overview](#overview)
  - [Install and run](#install-and-run)
  - [How to use this project](#how-to-use-this-project)
  - [Credits](#credits)
  - [License](#license)
  - [Testing](#testing)


## Overview

Partcile swarm optimization is a numerical optimization method that does not require knowing the exact gradient of the function being optimized. It is inspired by the behavior of real biological systems, such as flocks of birds. All of a sudden, it's very rare used in the context of control theory - so my goal was to explore its applicability for the control of a group of modile robots.
During this research, I've tryed different configuration of robots (so-called "swarm"): they can be totally connected to each other or have limited range of connection. Also robot's sensors can be broken so its measurement can be corrupted by some noise.

In the case you want to learn about the result, you can read my [report](./report.pdf) in Russian.

## Install and run

First on all, you need to clone this repo into your directory: 

```bash
git clone git@github.com:windowsartes/PSO4control.git
```

Then you need to install this package into your environment:

```bash
pip install .
```

After that, you will be able to use the cli:

```bash
python ./scene/scene.py path_to_your_config_file
```

Config file is a json-file, where you specify everything about current scene: the field, algorithm type and its hyperparameters, etc. You can find some examples in the [there](./config_examples), also I'll provide you a proper guide how to create this config file properly.


## How to use this project

This project was created with the goal to explore precision and robustness of PSO-based algorithms, so it the case you want to explore it more, for example, using different field or another vatiation of PSO-base algo, feel free to use this project as a base.

## Credits

I would like to thank my scientifique advisor, [Alexey S. Matveev](https://research.com/u/alexey-s-matveev) for helping me complete this work.

## License

I'm using the MIT license here, so feel free to use this project for any purpose.

## Testing

This project works compelety well on Windows 11; using Docker with WSL2, you will face a problem with GUI part of the application, but everything else works totally fine.