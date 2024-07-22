from abc import ABC, abstractmethod


class SolverInterface(ABC):
    @abstractmethod
    def turn(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def show(
        self,
        title: str,
    ) -> None:
        pass
