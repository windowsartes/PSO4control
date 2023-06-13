import typing as tp

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm


def gaussian(x: float, y: float, centre: tuple[float, float], sigma: float = 10):
    x0: float = centre[0]
    y0: float = centre[1]

    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma))


class Field:
    def __init__(self, height: float, width: float,
                 target_function: tp.Callable[[float, float, tuple[float, float]], float]):
        self._height = height
        self._width = width
        self._target_function = target_function

    @property
    def height(self) -> float:
        return self._height

    @property
    def width(self) -> float:
        return self._width

    @property
    def target_function(self) -> tp.Callable[[float, float, tuple[float, float]], float]:
        return self._target_function

    def show(self) -> None:
        x_values: np.ndarray = np.linspace(0, self.width, 1000)
        y_values: np.ndarray = np.linspace(0, self.height, 1000)

        centre: tuple[float, float] = (self.width/2, self.height/2)

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        coordinates = np.stack((x_grid.flatten(), y_grid.flatten()), -1)

        values = np.array([self.target_function(x, y, centre) for x, y in coordinates])
        values = values.reshape((len(x_values), len(y_values)))

        sns.heatmap(values, cmap=cm["hot"])
        plt.axis('off')

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x_grid, y_grid, values, cmap=cm["hot"],
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15)

        plt.show()


if __name__ == "__main__":
    field = Field(10, 10, gaussian)
    field.show()
