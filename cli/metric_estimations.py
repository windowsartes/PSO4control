import json
from pathlib import Path
from time import gmtime, strftime

import click
import numpy as np
import pandas as pd
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
    # spawns: list[str] = ["arc", "landing", "edge", "spot"]
    # noise_scale_values = [0.05, 0.075]
    with open("./logs.txt", "w", encoding="utf-8") as logs:
        logs.write(f'[{strftime("%Y-%m-%d %H:%M:%S", gmtime())}]: about to start\n')

    spawn_n_r_pairs: tuple[tuple[str, int, float], ...] = (
    )

    unique_n: tuple[int, ...] = tuple(set([n for _, n, _ in spawn_n_r_pairs]))
    unique_r: tuple[float, ...] = tuple(set([r for _, _, r in spawn_n_r_pairs]))
    unique_spawns: tuple[str, ...] = tuple(set([spawn for spawn, _, _ in spawn_n_r_pairs]))

    dataframes: dict[str, dict[str, pd.DataFrame]] = {}

    for spawn in unique_spawns:
        Path(f"./logs/{spawn}").mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframes[spawn] = {}

        names: tuple[str, str, str, str] = ("mear_error", "border_error", "mean_path", "border_path")

        for name in names:
            data: dict[str | int, tuple[float, ...]] = {}
            for n in unique_n:
                data[n] = tuple([-0.001] * len(unique_r))
            data["r"] = unique_r

            df = pd.DataFrame(data, dtype=float)
            df = df.set_index("r")

            dataframes[spawn][name] = df

        for name in names:
            dataframes[spawn][name].to_csv(
                f"./logs/{spawn}/{name}.tsv",
                sep="\t",
            )

    # for noise_scale in noise_scale_values:
    for spawn, n, r in spawn_n_r_pairs:
        with open(
            path_to_config,
            "r",
            encoding="utf-8",
        ) as config_file:
            config = json.load(config_file)
            # config["noise"]["params"]["scale"] = noise_scale
        config["solver"]["params"]["n_particles"] = n
        config["solver"]["params"]["connection_radius"] = r
        config["solver"]["params"]["spawn"]["type"] = spawn

        n_rooted: float = float(np.sqrt(config["solver"]["params"]["n_iterations"]))

        results = (Parallel(n_jobs=n_jobs)(
            delayed(Scene(config=config).solve)() for i in range(n_iterations))
        )

        error = np.array(results)[:, 0]
        path = np.array(results)[:, 1]

        mean_error: float = float(np.mean(error))
        border_error: float = float(2 * np.std(error, ddof=1) / n_rooted)

        mean_path: float = float(np.mean(path))
        border_path: float = float(2 * np.std(path, ddof=1) / n_rooted)

        dataframes[spawn]["mean_error"] = mean_error
        dataframes[spawn]["border_error"] = border_error
        dataframes[spawn]["mean_path"] = mean_path
        dataframes[spawn]["border_path"] = border_path

    for spawn in unique_spawns:
        names = ("mean_error", "border_error", "mean_path", "border_path")

        for name in names:
            dataframes[spawn][name].to_csv(
                f"./logs/{spawn}/{name}.tsv",
                sep="\t",
            )


if __name__ == "__main__":
    cli()  # noqa: E1120  # pylint: disable=no-value-for-parameter
