# Particle Swarm Optimization for control theory

- [Particle Swarm Optimization for control theory](#particle-swarm-optimization-for-control-theory)
  - [Overview](#overview)
  - [Install and run](#install-and-run)
    - [WSL/Windows integration](#wslwindows-integration)
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
python main.py path-to-config
```

Config file is a json-file, where you specify everything about current scene: the field, algorithm type and its hyperparameters, etc. You can find some examples [there](./config_examples), also I'll provide you a guide how to create this config file properly.

Also you can run this project with docker.

First off all you need to build an image using [Dockerfile](./Dockerfile).

```bash
docker build name:tag .
```

Then you can use constructed image to create a container using:

```bash
docker run -it --rm name:tag
```

### WSL/Windows integration

Firstly you need to set up everything according to [this guide](https://stackoverflow.com/questions/46018102/how-can-i-use-matplotlib-pyplot-in-a-docker-container).

Then you can create and run the container using:

```bash
docker run -it --rm -e DISPLAY=$DISPLAY name:tag
```


## How to use this project

This project was created with the goal to explore precision and robustness of PSO-based algorithms, so it the case you want to explore it more, for example, using different field or another vatiation of PSO-base algo, feel free to use this project as a base.

## Credits

I would like to thank my scientifique advisor, [Alexey S. Matveev](https://research.com/u/alexey-s-matveev) for helping me complete this work.

## License

I use the MIT license here, so feel free to use this project for any purpose.

## Testing

This project works compelety well on Windows 11: both native and WSL2. Also Docker integration works totally well.