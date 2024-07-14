import numpy as np
from pydantic import BaseModel


class Answer(BaseModel):
    answers: list[dict[str, float]]
