from src.field.field import FieldParameters, Field, AdditionalParameter
from src.field.target_function import TARGET_FUNCTION_REGISTER


class FieldFactory:
    @staticmethod
    def construct(config) -> Field:  # type: ignore
        field: Field = Field(
            FieldParameters(**config["params"]),
            None if config["additional_params"] is None else AdditionalParameter(**config["additional_params"]),
            TARGET_FUNCTION_REGISTER[config["type"].lower()],
        )

        return field


if __name__ == "__main__":
    config = {
        "params":
        {
            "size": 10.0,
            "quality_scale": 100.0,
            "centre": [5.0, 5.0],
            "sigma": 4.0
        },
        "type": "Rastrigin"
    }


    Factory = FieldFactory()

    field = Factory.construct(
        config
    )