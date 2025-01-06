import json
from pathlib import Path
from time import gmtime, strftime

import click
import numpy as np
import pandas as pd
import scipy.stats as sts
from joblib import Parallel, delayed

from src.scene.scene import Scene


@click.command()
@click.argument("path_to_config", type=click.Path(exists=True))
@click.argument("n_jobs", type=click.INT)
@click.argument("n_iterations", type=click.INT)
@click.argument("n_resamples", type=click.INT)
def cli(  # noqa: ER0914
    path_to_config: str,
    n_jobs: int,
    n_iterations: int,
    n_resamples: int,
) -> None:
    n_values: list[int] = [4, 5, 7, 10, 15]
    r_values: list[float] = [0.10, 0.25, 0.50, 1.0]
    spawns: list[str] = ["arc", "landing", "edge", "spot"]

    with open("./logs.txt", "w", encoding="utf-8") as logs:
        logs.write(f'[{strftime("%Y-%m-%d %H:%M:%S", gmtime())}]: about to start\n')

    for spawn in spawns:
        Path(f"./logs/{spawn}").mkdir(parents=True, exist_ok=True)

        dataframes: dict[str, pd.DataFrame] = {}
        names = ["normaltest", "shapiro", "cramer", "ks", "main_p_value"]

        for name in names:
            data = {}
            for n in n_values:
                data[n] = [-0.001] * len(r_values)
            data["r"] = r_values

            df = pd.DataFrame(data, dtype=float)
            df = df.set_index("r")

            dataframes[name] = df

        for n in n_values:
            for r in r_values:
                with open(path_to_config, "r", encoding="utf-8") as config_file:
                    config = json.load(config_file)
                    
                    config["solver"]["params"]["n_particles"] = n
                    config["solver"]["params"]["connection_radius"] = r
        
                    results = (Parallel(n_jobs=n_jobs)(delayed(Scene(config=config).solve)() for i in range(n_iterations)))
                    results = np.array(results)[:, 0]

                    bootstrapped_results = np.array([np.random.choice(results, results.shape[0], replace=True) for _ in range(n_resamples)])
                        
                    means = np.mean(bootstrapped_results, axis=1)

                    dataframes["normaltest"].loc[r, n] = round(
                        sts.normaltest(means).pvalue,
                        3
                    )

                    dataframes["shapiro"].loc[r, n] = round(
                        sts.shapiro(means).pvalue,
                        3
                    )

                    mean = np.mean(means)
                    std = np.std(means, ddof=1)

                    dataframes["cramer"].loc[r, n] = round(
                        sts.cramervonmises(means, cdf=sts.norm.cdf, args=(mean, std)).pvalue,
                        3
                    )

                    dataframes["ks"].loc[r, n] = round(
                        sts.ks_1samp(means, cdf=sts.norm.cdf, args=(mean, std)).pvalue,
                        3
                    )

                    if spawn in ["arc", "landing"]:
                        p: float = 0.17580727
                    else:
                        p = 0.237625

                    dataframes["main_p_value"].loc[r, n] = round(
                        sts.ttest_1samp(results, popmean=p * 0.4, alternative="less").pvalue,
                        3
                    )

                    with open("./logs.txt", "a", encoding="utf-8") as logs:
                        logs.write(
                            f'[{strftime("%Y-%m-%d %H:%M:%S", gmtime())}]:' +
                            f'ended with spawn = {spawn} | n = {n} | r = {r}\n',
                        )

        for name in names:
            dataframes[name].to_csv(f"./logs/{spawn}/{name}.tsv", sep="\t")


if __name__ == "__main__":
    cli()  # noqa: E1120
