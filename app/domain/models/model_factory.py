from typing import Dict, Type
from app.domain.models.abstract_model import AbstractModel
from app.domain.models.post_only_model import PostOnlyModel
from app.domain.models.post_premask_model import PostPremaskModel
from app.config import WEIGHTS_MAP, HF_REPO_ID
from app.domain.models.pre_post_model import PrePostModel
from app.domain.models.pre_post_premask_model import PrePostPremaskModel
import streamlit as st

class ModelFactory:
    _instances: Dict[str, AbstractModel] = {}

    _model_registry: Dict[str, Type[AbstractModel]] = {
        "post_only": PostOnlyModel,
        "post_premask": PostPremaskModel,
        "pre_post": PrePostModel,
        "pre_post_premask": PrePostPremaskModel
    }

    @staticmethod
    @st.cache_resource
    def download_and_load_model(model_type, model_class):
        print(f"--- Downloading and loading {model_type} model... ---")
        filename = WEIGHTS_MAP.get(model_type)

        weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, token=st.secrets.get("HF_TOKEN"))

        instance = model_class(weights_path)
        instance.load_weights()
        return instance

    @classmethod
    def get_model(cls, model_type: str) -> AbstractModel:
        if model_type not in cls._model_registry:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        instance_key = f"{model_type}"

        if instance_key not in cls._instances:
            print(f"--- Loading {model_type} model into memory... ---")
            model_class = cls._model_registry[model_type]

            # weights_path = WEIGHTS_MAP.get(model_type)
            # instance = model_class(weights_path)
            # instance.load_weights()
            #
            # cls._instances[instance_key] = instance

        #return cls._instances[instance_key]
        return ModelFactory.download_and_load_model(model_type,model_class)

