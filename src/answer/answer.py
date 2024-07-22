from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float
    value: float


class Answer(BaseModel):
    answers: list[Point]
