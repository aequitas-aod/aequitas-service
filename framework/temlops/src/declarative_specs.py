import yaml
from data_preparation import *
from inspect import getmembers, isfunction, ismethod, isclass


class DataPreparationSpecs:
    def __init__(self):
        pass

    def load_data_configs(self):
        pass

    def save_data_config(self):
        pass


class ModellingSpecs:
    def __init__():
        pass

    def load_modelling_config(self):
        pass

    def save_modelling_config(self):
        pass


def class_to_dict(cls) -> dict:
    """Convert a class's attributes and methods to a dictionary."""
    spec = {
        "class_name": cls.__class__.__name__,
        "docstring": cls.__doc__,
        "attributes": {},
        "methods": {},
        "static_methods": {},
    }

    # Get all members (attributes/methods) of the class
    for name, value in getmembers(cls):
        # Skip private members and built-ins
        if name.startswith("_"):
            continue

        # Handle attributes
        if not (isfunction(value) or ismethod(value) or isclass(value)):
            spec["attributes"][name] = value

        # Handle methods
        elif ismethod(value) or isfunction(value):
            if isinstance(value, staticmethod):
                spec["static_methods"][name] = {
                    "docstring": value.__doc__,
                    "signature": str(value.__annotations__),
                }
            else:
                spec["methods"][name] = {
                    "docstring": value.__doc__,
                    "signature": str(value.__annotations__),
                }

    return spec


if __name__ == "__main__":
    dt_prep = DataPreparation()

    # Get class specifications as a dictionary
    class_spec = class_to_dict(dt_prep)

    # Save to YAML
    with open("metadata/data_preparation_specs.yaml", "w") as f:
        yaml.dump(class_spec, f, default_flow_style=False, sort_keys=False)

    print(
        "DataPreparation specifications saved to 'metadata/data_preparation_specs.yaml'"
    )
