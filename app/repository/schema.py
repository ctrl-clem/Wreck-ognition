from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import torch

@dataclass
class ModelInput:
    post_image: torch.Tensor
    pre_image:  torch.Tensor | None
    pre_mask:   torch.LongTensor | None


@dataclass
class PredictionResult:
    post_mask: np.ndarray
    gradcam_post: np.ndarray
    colored_overlay: Image.Image
    class_distribution: dict[str, float]
    confidence_scores: dict[str,float]
    damage_density: float
    latency: float
    logits: torch.Tensor
    model_name: str = ""
