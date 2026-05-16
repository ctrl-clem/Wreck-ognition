from dataclasses import dataclass, field
from typing import Dict, List, Optional
from PIL import Image


@dataclass
class EntropyArtifact:
    name: str
    score: float
    image: Image.Image


@dataclass
class ModelArtifacts:
    model_name: str
    inference_time: float
    damage_density: float
    prediction_overlay: Image.Image
    distribution_graph: Image.Image
    confidence_levels: Dict[str, float]
    gradcam_image: Image.Image
    entropies: List[EntropyArtifact]


@dataclass
class ComparisonArtifacts:
    magnitude_maps: Dict[str, Image.Image]
    targeted_disagreements: List[Dict]


@dataclass
class FullReportData:
    project_id: str
    pre_image: Image.Image
    post_image: Image.Image
    models: Dict[str, ModelArtifacts]
    comparison: Optional[ComparisonArtifacts] = None
    common_legends: Dict[str, Image.Image] = field(default_factory=dict)