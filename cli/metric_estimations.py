import json
from time import gmtime, strftime

import click
import numpy as np
from joblib import Parallel, delayed

from src.scene.scene import Scene


@click.command()
@click.argument("path_to_config", type=click.Path(exists=True))
@click.argument("n_jobs", type=click.INT)
@click.argument("n_iterations", type=click.INT)
def cli(
    path_to_config: str,
    n_jobs: int,
    n_iterations: int,
) -> None:
    n_r_strategy_pairs: tuple[tuple[int, float, str], ...] = (
        (10, 0.1, "spot"),
    )

    with open(
        path_to_config,
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    logs_file = f"./results/{config['field']['type'].lower()}_{config['solver']['specification']}.txt"

    with open(
        logs_file,
        "w",
        encoding="utf-8",
    ) as logs:
        logs.write(f'[{strftime("%Y-%m-%d %H:%M:%S", gmtime())}]: about to start\n')

    for n, r, strategy in n_r_strategy_pairs:
        with open(
            path_to_config,
            "r",
            encoding="utf-8",
        ) as config_file:
            config = json.load(config_file)

        if config['solver']['type'] == "swarm":
            config["solver"]["params"]["n_particles"] = n
        elif config['solver']['type'] == "grey_wolf_optimization":
            config["solver"]["params"]["n_wolves"] = n
        elif config['solver']['type'] == "artificial_bee_colony":
            config["solver"]["params"]["n_bees"] = n
            config["solver"]["params"]["source_limit"] = n - n // 2

        config["solver"]["params"]["connection_radius"] = r
        config["solver"]["params"]["spawn"]["type"] = strategy

        if config['field']['type'] == "Griewank" or config['field']['type'] == "Rastrigin":
            config["field"]["additional_params"] = None

        n_rooted: float = float(np.sqrt(n_iterations))

        results = (Parallel(n_jobs=n_jobs)(
            delayed(Scene(config=config).solve)() for i in range(n_iterations))
        )

        error = np.array(results)[:, 0]
        path = np.array(results)[:, 1]

        mean_error: float = round(float(np.mean(error)), 3)
        border_error: float = round(float(2 * np.std(error, ddof=1) / n_rooted), 3)

        mean_path: float = round(float(np.mean(path)), 2)
        border_path: float = round(float(2 * np.std(path, ddof=1) / n_rooted), 2)

        with open(
            logs_file,
            "a",
            encoding="utf-8",
        ) as logs:
            logs.write(
                f'[{strftime("%Y-%m-%d %H:%M:%S", gmtime())}]: ' +\
                f'ended with strategy = {strategy} | n = {n} | r = {r}; ' +\
                f'error: {mean_error} +/- {border_error}; path: {mean_path} +/- {border_path}\n'
            )

    with open(
        logs_file,
        "a",
        encoding="utf-8",
    ) as logs:
        logs.write(
            f'[{strftime("%Y-%m-%d %H:%M:%S", gmtime())}]: ended',
        )


if __name__ == "__main__":
    cli()
