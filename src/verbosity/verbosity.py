from dataclasses import dataclass


@dataclass
class Verbosity:
    value: int
    period: int
