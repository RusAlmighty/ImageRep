from pydantic import BaseModel


def add_model(parser, model):
    "Add Pydantic model to an ArgumentParser"
    fields = model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.type_,
            default=field.default,
            help=field.field_info.description,
        )


class ModelConfig(BaseModel):
    name: str = "exp1"
    in_features: int = 3
    out_features: int = 3
    hidden_features: int = 1024
    hidden_layers: int = 9
    outermost_linear: bool = True
    first_omega_0: int = 30
    hidden_omega_0: int = 60

    drop_out: float = 0.25
    train_workers: int = 0
    limit: int = 1000
    batch_size: int = 16
    lr: float = 1E-5
    wd: float = 1E-2
    schedule_step: int = 400
    cach: bool = True
    set_transparent_to_white: bool = True

    train_dataset_path: str = r"Data/96"
    validation_dataset_path: str = "Data/256"
