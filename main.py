import json

import click
from joblib import Parallel, delayed
from tqdm import tqdm

from src.scene.scene import Scene


@click.command()
@click.argument("path_to_config", type=click.Path(exists=True))
@click.argument("n_jobs", type=click.INT)
@click.argument("n_iterations", type=click.INT)
def cli(
    path_to_config: click.Path(exists=True),
    n_jobs: click.INT,
    n_iterations: click.INT,
) -> None:
    # print(scene.solve())
    noise_scale_values = [0.001, 0.0025, 0.02, 0.05, 0.075]
    n_values = [3, 4, 5, 10, 15, 20]
    r_values = [0.05, 0.10, 0.25, 0.50, 1.0]

    total = len(noise_scale_values) * len(n_values) * len(r_values)

    i = 1
    for noise_scale in noise_scale_values:
        for n in n_values:
            for r in r_values:
                with open(path_to_config, "r") as config_file:
                    config = json.load(config_file)
                    config["noise"]["params"]["scale"] = noise_scale
                    config["solver"]["params"]["n_particles"] = n
                    config["solver"]["params"]["connection_radius"] = r

                with open(f"./{config['solver']['params']['spawn']['type']}_{n}_{r}_{noise_scale * 100}.csv", 'w') as f: 
                    results = (Parallel(n_jobs=n_jobs)(delayed(Scene(config=config).solve)() for i in range(n_iterations)))
                    f.write('error,path\n')
                    for (error, path) in results:
                        f.write(f'{error},{path}\n')
                print(f'{i}/{total} done')
                i+=1


if __name__ == "__main__":
    cli()
