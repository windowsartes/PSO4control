import json

import click
import numpy as np
import scipy.stats as sts
from joblib import Parallel, delayed
from tqdm import tqdm

from src.scene.scene import Scene


@click.command()
@click.argument("path_to_config", type=click.Path(exists=True))
@click.argument("n_jobs", type=click.INT)
@click.argument("n_iterations", type=click.INT)
@click.argument("n_resamples", type=click.INT)
def cli(
    path_to_config: click.Path(exists=True),
    n_jobs: click.INT,
    n_iterations: click.INT,
    n_resamples: click.INT,
) -> None:
    # print(scene.solve())
    
    with open(path_to_config, "r") as config_file:
        config = json.load(config_file)

    results = (Parallel(n_jobs=-1)(delayed(Scene(config=config).solve)() for i in range(n_iterations)))
    results = np.array(results)[:, 0]
        
    bootstrapped_results = np.array([np.random.choice(results, results.shape[0], replace = True) for _ in range(n_resamples)])
        
    means = np.mean(bootstrapped_results, axis = 1)

    print(sts.normaltest(means).pvalue)
    print(sts.shapiro(means).pvalue)

    mean = np.mean(means)
    std = np.std(means, ddof=1)

    print(sts.cramervonmises(means, cdf = sts.norm.cdf, args=(mean, std)).pvalue)
    print(sts.ks_1samp(means, cdf = sts.norm.cdf, args=(mean, std)).pvalue)

    print('-----------------------')

    print(np.mean(results))

    # p = 0.15693176
    p = 0.07532275

    print(sts.ttest_1samp(results, popmean = p * 0.4, alternative = "less").pvalue)

if __name__ == "__main__":
    cli()
