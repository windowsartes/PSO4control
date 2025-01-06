from src.field.field import FieldParameters, Field, AdditionalParameter
from src.field.target_function import TARGET_FUNCTION_REGISTER  # , TARGET_FUNCTION_SYMBOLIC_REGISTER


class FieldFactory:
    @staticmethod
    def construct(config) -> Field:  # type: ignore
        field: Field = Field(
            FieldParameters(**config["params"]),
            None if "additional_params" not in config else AdditionalParameter(**config["additional_params"]),
            TARGET_FUNCTION_REGISTER[config["type"].lower()],
            # TARGET_FUNCTION_SYMBOLIC_REGISTER[config["type"].lower()],
        )

        return field
