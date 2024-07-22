import click

from src.scene.scene import Scene


@click.command()
@click.argument("path_to_config", type=click.Path(exists=True))
def cli(path_to_config: click.Path(exists=True)) -> None:
    scene: Scene = Scene(path_to_config)
    scene.solve()


if __name__ == "__main__":
    cli()
