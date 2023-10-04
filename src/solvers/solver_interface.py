from abc import ABC, abstractmethod


class SolverInterface(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def show_current_position(self, *args, **kwargs):
        pass
