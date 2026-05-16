import torch
from PIL import Image
from app.domain.models.model_factory import ModelFactory
from app.repository.schema import ModelInput, PredictionResult
from app.repository.preprocessing import Preprocessing
import numpy as np
from app.config import CLASS_LABELS, CLASS_COLORS
import time
from pytorch_grad_cam import GradCAM
from app.domain.damage_model_architecture.architecture import DamageModel
from app.domain.models.abstract_model import AbstractModel
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from app.service.utils.gradcam_wrapper import MultiInputCamWrapper
import itertools
import streamlit as st
from streamlit import cache_resource
import matplotlib.pyplot as plt
import torch
import streamlit as st

class InferenceService():
    def __init__(self):
        self.factory = ModelFactory()


    def run_inference(_self, model_type: str, pre_img, post_img, premask):
        model = _self.factory.get_model(model_type)
        model_input = _self._prepare_input(pre_img, post_img, premask)
        start_time = time.perf_counter()
        logits,features = model.predict(model_input)
        end_time = time.perf_counter()
        latency_seconds = round(end_time - start_time, 3)
        return _self.build_result(model_type,model, model_input,logits, latency_seconds)

    def _prepare_input(self, pre_img, post_img, premask) -> ModelInput:
        return Preprocessing.build_model_input(pre=pre_img,post=post_img,premask=premask)


    def build_result(self, model_type,model, model_input, logits: torch.Tensor, latency) -> PredictionResult:
        mask_np, conf_np = self.extract_predictions(logits)
        class_distribution, confidence_score = self.calculate_class_metrics(mask_np, conf_np)
        damage_density = self.calculate_damage_density(mask_np)
        colored_overlay = self.create_colored_overlay(mask_np)
        #gradcam = self.compute_gradcam_post_image(model=model, model_input=model_input)

        return PredictionResult(
            post_mask=mask_np,
            colored_overlay = colored_overlay,
            class_distribution = class_distribution,
            confidence_scores= confidence_score,
            damage_density=damage_density,
            latency=latency,
            #gradcam_post=gradcam,
            gradcam_post=None,
            logits=logits,
            model_name=model_type
        )

    def calculate_class_metrics(self, mask_np, conf_np):
        total_pixels = mask_np.size
        class_distribution = {}
        confidence_scores = {}

        for label_idx, label_name in CLASS_LABELS.items():
            class_mask = (mask_np == label_idx)
            pixel_count = np.sum(class_mask)

            class_distribution[label_name] = pixel_count

            if pixel_count > 0:
                class_confidences = conf_np[class_mask]
                avg_conf = np.mean(class_confidences)
                confidence_scores[label_name] = round(float(avg_conf * 100.0), 2)
            else:
                confidence_scores[label_name] = 0.0

        return class_distribution, confidence_scores

    def calculate_damage_density(self, mask_np):
        undamaged_pixels = 0
        damaged_pixels = 0

        for label_idx, label_name in CLASS_LABELS.items():
            name_lower = label_name.lower()

            if "background" in name_lower:
                continue

            pixel_count = np.sum(mask_np == label_idx)

            if "no-damage" in name_lower:
                undamaged_pixels += pixel_count
            else:
                damaged_pixels += pixel_count

        total_building_pixels = undamaged_pixels + damaged_pixels

        if total_building_pixels == 0:
            return 0.0

        density = (damaged_pixels / total_building_pixels) * 100.0
        return round(float(density), 2)

    def create_colored_overlay(self, mask_np: np.ndarray) -> Image.Image:
        colored_mask_np = CLASS_COLORS[mask_np]
        alpha_channel = np.full(mask_np.shape, 180, dtype=np.uint8)

        alpha_channel[mask_np == 0] = 0

        rgba_mask_np = np.dstack((colored_mask_np, alpha_channel))

        return Image.fromarray(rgba_mask_np, mode="RGBA")

        #colored_mask_np = colored_mask_np.astype(np.uint8)
        #return Image.fromarray(colored_mask_np)


    def extract_predictions(self, logits: torch.Tensor):
        probabilities = torch.softmax(logits, dim=1)
        max_probs, predicted_classes = torch.max(probabilities, dim=1)

        mask_np = predicted_classes.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        conf_np = max_probs.squeeze(0).detach().cpu().numpy()

        return mask_np, conf_np

    def compute_gradcam_post_image(self, model, model_input):
        pytorch_model = model.get_damage_model()
        wrapped_model = MultiInputCamWrapper(pytorch_model)
        mode = pytorch_model.mode


        target_layers = [wrapped_model.model.up1]

        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        if mode == 'post_only':
            tensor_list = [model_input.post_image]
        elif mode == 'post_premask':
            tensor_list = [model_input.post_image, model_input.pre_mask]
        elif mode == 'pre_post':
            tensor_list = [model_input.pre_image, model_input.post_image]
        elif mode == 'pre_post_premask':
            tensor_list = [model_input.pre_image, model_input.post_image, model_input.pre_mask]
        else:
            tensor_list = [model_input.post_image]

        input_tensor = torch.cat(tensor_list, dim=1)

        with torch.no_grad():
            logits = wrapped_model(input_tensor)

        pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

        cams = []

        for class_idx in range(1, 5):
            mask = (pred_mask == class_idx)

            if not mask.any():
                continue

            target = SemanticSegmentationTarget(class_idx, mask)

            grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
            cams.append(grayscale_cam[0, :])

        if not cams:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        grayscale_cam = np.mean(cams, axis=0)

        return grayscale_cam

    def calculate_model_disagreement(self, result_a: PredictionResult, result_b: PredictionResult):
        mask_a = result_a.post_mask
        mask_b = result_b.post_mask


        building_mask = (mask_a != 0) | (mask_b != 0)

        disagreement_mask = (mask_a != mask_b) & building_mask

        total_building_pixels = np.sum(building_mask)
        if total_building_pixels == 0:
            disagreement_rate = 0.0
        else:
            disagreement_rate = (np.sum(disagreement_mask) / total_building_pixels) * 100.0


        h, w = mask_a.shape
        rgba_disagreement = np.zeros((h, w, 4), dtype=np.uint8)

        rgba_disagreement[disagreement_mask] = [255, 0, 255, 200]

        return Image.fromarray(rgba_disagreement, mode="RGBA"), round(float(disagreement_rate), 2)

    def calculate_directional_disagreement(self, result_a, result_b):
        mask_a = result_a.post_mask.astype(np.int8)
        mask_b = result_b.post_mask.astype(np.int8)

        building_mask = (mask_a != 0) | (mask_b != 0)

        diff = mask_b - mask_a

        h, w = mask_a.shape
        rgba_diff = np.zeros((h, w, 4), dtype=np.uint8)

        rgba_diff[(diff > 0) & building_mask] = [255, 50, 50, 200]

        rgba_diff[(diff < 0) & building_mask] = [50, 150, 255, 200]

        return Image.fromarray(rgba_diff, mode="RGBA")

    def calculate_targeted_disagreement(self, result_a, result_b, class_a_idx: int, class_b_idx: int):
        mask_a = result_a.post_mask
        mask_b = result_b.post_mask

        target_mask = (mask_a == class_a_idx) & (mask_b == class_b_idx)

        h, w = mask_a.shape
        rgba_target = np.zeros((h, w, 4), dtype=np.uint8)

        rgba_target[target_mask] = [0, 255, 255, 200]

        building_pixels = np.sum((mask_a != 0) | (mask_b != 0))
        rate = (np.sum(target_mask) / building_pixels) * 100.0 if building_pixels > 0 else 0.0

        return Image.fromarray(rgba_target, mode="RGBA"), round(float(rate), 2)



    def calculate_entropy_metrics(self, logits: torch.Tensor, premask_np: np.ndarray):
        probs = torch.softmax(logits, dim=1)
        epsilon = 1e-9

        entropy_tensor = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
        normalized_entropy = entropy_tensor / np.log(probs.shape[1])

        entropy_np = normalized_entropy.squeeze(0).detach().cpu().numpy()

        building_indices = premask_np > 0

        if np.any(building_indices):
            total_building_entropy = np.mean(entropy_np[building_indices])
        else:
            total_building_entropy = 0.0


        cm = plt.get_cmap('magma')
        colored_entropy = cm(entropy_np)  # Returns RGBA [0, 1]

        heatmap_img = Image.fromarray((colored_entropy[:, :, :3] * 255).astype(np.uint8))

        return heatmap_img, round(float(total_building_entropy), 4)

    def _get_raw_entropy_array(self, logits: torch.Tensor) -> np.ndarray:
        s = 30
        probs = torch.softmax(logits*s, dim=1)
        epsilon = 1e-9
        # Shannon Entropy normalized to [0, 1]
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
        normalized_entropy = entropy / np.log(probs.shape[1])
        return normalized_entropy.squeeze(0).detach().cpu().numpy()

    def compute_agreement_entropy(self, logits, premask_np):
        """Buildings in BOTH prediction and ground truth."""
        entropy_map = self._get_raw_entropy_array(logits)
        preds = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        mask = (preds > 0) & premask_np

        score = np.mean(entropy_map[mask]) if np.any(mask) else 0.0
        # Return only the entropy belonging to the mask
        final_map = np.where(mask, entropy_map, 0.0)
        return final_map, round(float(score), 4)

    def compute_hallucination_entropy(self, logits, premask_np):
        """Building in prediction but NOT in ground truth."""
        entropy_map = self._get_raw_entropy_array(logits)
        preds = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        mask = (preds > 0) & (~premask_np)  # ~ is NOT

        score = np.mean(entropy_map[mask]) if np.any(mask) else 0.0
        final_map = np.where(mask, entropy_map, 0.0)
        return final_map, round(float(score), 4)

    def compute_omission_entropy(self, logits, premask_np):
        """Building in ground truth but NOT in prediction."""
        entropy_map = self._get_raw_entropy_array(logits)
        preds = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        mask = (preds == 0) & premask_np

        score = np.mean(entropy_map[mask]) if np.any(mask) else 0.0
        final_map = np.where(mask, entropy_map, 0.0)
        return final_map, round(float(score), 4)

    def convert_premask_to_np(self,premask):
        if hasattr(premask, 'convert'):
            premask_img_raw = premask.convert('L')
        else:
            premask_img_raw = Image.open(premask).convert('L')

        premask_img = Preprocessing.resize(premask_img_raw, resampling_mode='NEAREST')

        premask_np = np.array(premask_img)
        return premask_np > 0



