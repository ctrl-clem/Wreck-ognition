import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
from app.service.inference_service import InferenceService
from app.repository.schema import PredictionResult


@pytest.fixture
def service():
    with patch("app.service.inference_service.ModelFactory"):
        return InferenceService()


@pytest.fixture
def dummy_logits():
    return torch.rand(1, 5, 10, 10)


@pytest.fixture
def dummy_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, 0:5] = 1
    mask[5:10, 5:10] = 4
    return mask


@pytest.fixture
def dummy_prediction_result():
    def _make_result(mask_array):
        result = MagicMock(spec=PredictionResult)
        result.post_mask = mask_array
        return result

    return _make_result



def test_extract_predictions(service, dummy_logits):
    mask_np, conf_np = service.extract_predictions(dummy_logits)

    assert mask_np.shape == (10, 10)
    assert conf_np.shape == (10, 10)
    assert mask_np.dtype == np.uint8


@patch("app.service.inference_service.CLASS_LABELS", {0: "Background", 1: "No-Damage", 4: "Destroyed"})
def test_calculate_class_metrics(service, dummy_mask):
    conf_np = np.full((10, 10), 0.8, dtype=np.float32)

    distribution, confidences = service.calculate_class_metrics(dummy_mask, conf_np)

    assert distribution["Background"] == 50
    assert distribution["No-Damage"] == 25
    assert distribution["Destroyed"] == 25

    assert confidences["No-Damage"] == 80.0
    assert confidences["Destroyed"] == 80.0


@patch("app.service.inference_service.CLASS_LABELS",
       {0: "Background", 1: "No-Damage", 2: "Minor-Damage", 4: "Destroyed"})
def test_calculate_damage_density(service, dummy_mask):
    density = service.calculate_damage_density(dummy_mask)
    assert density == 50.0


@patch("app.service.inference_service.CLASS_COLORS", np.array([
    [0, 0, 0],  # 0: Background
    [0, 255, 0],  # 1: No-Damage
    [255, 255, 0],  # 2: Minor
    [255, 165, 0],  # 3: Major
    [255, 0, 0]  # 4: Destroyed
], dtype=np.uint8))
def test_create_colored_overlay(service, dummy_mask):
    img = service.create_colored_overlay(dummy_mask)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"
    assert img.size == (10, 10)

    bg_pixel = img.getpixel((9, 0))
    assert bg_pixel[3] == 0

    building_pixel = img.getpixel((0, 0))
    assert building_pixel == (0, 255, 0, 180)


def test_calculate_model_disagreement(service, dummy_prediction_result):
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_b = np.zeros((10, 10), dtype=np.uint8)

    mask_a[0:5, 0:5] = 1
    mask_b[0:5, 0:5] = 2

    res_a = dummy_prediction_result(mask_a)
    res_b = dummy_prediction_result(mask_b)

    img, rate = service.calculate_model_disagreement(res_a, res_b)

    assert rate == 100.0
    assert isinstance(img, Image.Image)


def test_calculate_targeted_disagreement(service, dummy_prediction_result):
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_b = np.zeros((10, 10), dtype=np.uint8)

    mask_a[0:2, 0:2] = 1
    mask_b[0:2, 0:2] = 4

    mask_a[5:7, 5:7] = 2
    mask_b[5:7, 5:7] = 3

    res_a = dummy_prediction_result(mask_a)
    res_b = dummy_prediction_result(mask_b)

    img, rate = service.calculate_targeted_disagreement(res_a, res_b, 1, 4)

    assert rate == 50.0


def test_convert_premask_to_np(service):
    img = Image.new("L", (10, 10), color=0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 4, 4], fill=255)

    with patch("app.service.inference_service.Preprocessing.resize", return_value=img):
        np_mask = service.convert_premask_to_np(img)

        assert np_mask.shape == (10, 10)
        assert np_mask.dtype == bool
        assert np_mask[0, 0] == True
        assert np_mask[9, 9] == False


def test_entropy_metrics_math(service, dummy_logits):
    premask = np.zeros((10, 10), dtype=bool)
    premask[0:5, 0:5] = True

    heatmap_img, score = service.calculate_entropy_metrics(dummy_logits, premask)

    assert isinstance(heatmap_img, Image.Image)
    assert heatmap_img.mode == "RGB"
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0