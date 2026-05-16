from app.domain.models.abstract_model import AbstractModel
from app.domain.damage_model_architecture.architecture import DamageModel
from app.repository.schema import ModelInput, PredictionResult
from app.config import CLASS_LABELS, CLASS_COLORS
import torch
import numpy as np
from PIL import Image

class PrePostPremaskModel(AbstractModel):
    def __init__(self,weights_path):
        self.name = "pre_post_premask"
        self.required_inputs = ["pre","post","premask"]
        self.weights_path = weights_path
        self.weights = {}
        self.damage_model = DamageModel(mode=self.name,num_classes=5)


    def load_weights(self) -> None:
        state_dict = torch.load(self.weights_path, map_location=torch.device('cpu'))
        self.damage_model.load_state_dict(state_dict)
        self.damage_model.eval()


    def predict(self, inputs:ModelInput):
        pre_tensor = inputs.pre_image
        pre_mask = inputs.pre_mask
        post_tensor = inputs.post_image

        with torch.no_grad():
            logits, features = self.damage_model(pre_tensor,post_tensor,pre_mask)

        return logits, features

