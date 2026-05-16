from app.repository.schema import ModelInput, PredictionResult
from abc import ABC, abstractmethod
from app.domain.damage_model_architecture.architecture import DamageModel
import torch
import numpy as np
from app.config import CLASS_LABELS, CLASS_COLORS
from PIL import Image
class AbstractModel(ABC):

    name : str
    weights_path : str
    required_inputs : list[str]
    weights : dict
    damage_model : DamageModel

    @abstractmethod
    def load_weights(self) -> None: ...

    @abstractmethod
    def predict(self, inputs: ModelInput): ...

    def get_damage_model(self):
        return self.damage_model





