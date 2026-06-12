import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
from reportlab.platypus import Image as RLImage, Paragraph, Table
from app.service.report_service import ReportService

class MockEntropy:
    def __init__(self, name, score, image):
        self.name = name
        self.score = score
        self.image = image


class MockArtifact:
    def __init__(self, img):
        self.inference_time = 1.25
        self.damage_density = 45.5
        self.prediction_overlay = img
        self.distribution_graph = img
        self.gradcam_image = img
        self.entropies = [
            MockEntropy("High Certainty", 0.9, img),
            MockEntropy("Low Certainty", 0.4, img)
        ]


class MockReportData:
    def __init__(self, img):
        self.project_id = "TEST-PROJECT-123"
        self.pre_image = img
        self.post_image = img
        self.models = {
            "resnet50": MockArtifact(img),
            "efficientnet": MockArtifact(img)
        }


@pytest.fixture
def service():
    return ReportService()


@pytest.fixture
def dummy_pil_image():
    return Image.new("RGB", (10, 10), color="black")


@pytest.fixture
def dummy_numpy_image():
    return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def report_styles():
    from reportlab.lib.styles import getSampleStyleSheet
    styles = getSampleStyleSheet()
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    styles.add(ParagraphStyle('Caption', parent=styles['Italic'], alignment=TA_CENTER, fontSize=9))
    return styles



def test_process_image_to_flowable_with_numpy(service, dummy_numpy_image):
    flowable = service.process_image_to_flowable(dummy_numpy_image)
    assert isinstance(flowable, RLImage)
    # Check default width and height constraints
    assert flowable.drawWidth == 72 * 3.2  # 3.2 inches * 72 points
    assert flowable.drawHeight == 72 * 3.2


def test_process_image_to_flowable_with_pil(service, dummy_pil_image):
    flowable = service.process_image_to_flowable(dummy_pil_image)
    assert isinstance(flowable, RLImage)


def test_process_image_to_flowable_none(service):
    flowable = service.process_image_to_flowable(None)
    assert isinstance(flowable, Paragraph)
    assert flowable.getPlainText() == "No Image Available"


def test_process_image_to_flowable_matplotlib(service):
    mock_fig = MagicMock()
    mock_fig.savefig = MagicMock()

    with patch("app.service.report_service.Image.open") as mock_open:
        mock_open.return_value = Image.new("RGB", (10, 10))
        flowable = service.process_image_to_flowable(mock_fig)

        assert mock_fig.savefig.called
        assert isinstance(flowable, RLImage)


def test_create_dual_image_table(service, dummy_pil_image, report_styles):
    table = service.create_dual_image_table(
        dummy_pil_image, "Left Caption",
        dummy_pil_image, "Right Caption",
        report_styles
    )

    assert isinstance(table, Table)

    cell_data = table._cellvalues
    assert len(cell_data) == 2
    assert len(cell_data[0]) == 2

    assert isinstance(cell_data[0][0], RLImage)
    assert isinstance(cell_data[0][1], RLImage)

    assert isinstance(cell_data[1][0], Paragraph)
    assert cell_data[1][0].getPlainText() == "Left Caption"


def test_add_header_footer(service):
    mock_canvas = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page = 1

    service.add_header_footer(mock_canvas, mock_doc)

    mock_canvas.saveState.assert_called_once()
    mock_canvas.restoreState.assert_called_once()

    assert mock_canvas.drawCentredString.call_count == 2
    calls = mock_canvas.drawCentredString.call_args_list

    assert calls[0][0][2] == "Ablation Study: Disaster Damage Assessment Report"
    assert calls[1][0][2] == "Page 1"


def test_generate_pdf_integration(service, dummy_pil_image):
    mock_data = MockReportData(dummy_pil_image)

    pdf_bytes = service.generate_pdf(mock_data)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF-")