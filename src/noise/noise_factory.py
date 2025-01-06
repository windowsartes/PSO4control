from src.answer.answer import Answer
from src.noise.noise import NOISE_REGISTER, NoiseBase, NoiseHyperparameters


class NoiseFactory:
    @staticmethod
    def construct(answer: Answer, config) -> NoiseBase:  # type: ignore
        noise_hyperparameters: NoiseHyperparameters = NoiseHyperparameters(**config["params"])

        return NOISE_REGISTER[config["type"].lower()](answer, noise_hyperparameters)
