from src.field.field import FieldParameters, Field
from src.field.target_function import TARGET_FUNCTION_REGISTER, TARGET_FUNCTION_SYMBOLIC_REGISTER


class FieldFactory:
    def construct(self, field_config) -> Field:
        field: Field = Field(
            FieldParameters(**field_config["params"]),
            TARGET_FUNCTION_REGISTER[field_config["type"].lower()],
            TARGET_FUNCTION_SYMBOLIC_REGISTER[field_config["type"].lower()],
        )

        return field
