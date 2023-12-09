# import local libraries
from src.model import ModelFactory
from src.transformation import TransformFactory


class ModelTransformFactory:
    def __init__(self):
        pass

    def __call__(self, model_name, cfg):
        train_transform, val_transform = TransformFactory(cfg)(model_name)
        model = ModelFactory(cfg)(model_name)
        return {
            "model": model,
            "train_transform": train_transform,
            "val_transform": val_transform,
        }
