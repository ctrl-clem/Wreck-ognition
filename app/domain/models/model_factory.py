from typing import Dict, Type
from app.domain.models.abstract_model import AbstractModel
from app.domain.models.post_only_model import PostOnlyModel
from app.domain.models.post_premask_model import PostPremaskModel
from app.config import WEIGHTS_MAP
from app.domain.models.pre_post_model import PrePostModel
from app.domain.models.pre_post_premask_model import PrePostPremaskModel

class ModelFactory:
    _instances: Dict[str, AbstractModel] = {}

    _model_registry: Dict[str, Type[AbstractModel]] = {
        "post_only": PostOnlyModel,
        "post_premask": PostPremaskModel,
        "pre_post": PrePostModel,
        "pre_post_premask": PrePostPremaskModel
    }

    @classmethod
    def get_model(cls, model_type: str) -> AbstractModel:
        if model_type not in cls._model_registry:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        instance_key = f"{model_type}"

        if instance_key not in cls._instances:
            print(f"--- Loading {model_type} model into memory... ---")
            model_class = cls._model_registry[model_type]

            weights_path = WEIGHTS_MAP.get(model_type)
            instance = model_class(weights_path)
            instance.load_weights()

            cls._instances[instance_key] = instance

        return cls._instances[instance_key]