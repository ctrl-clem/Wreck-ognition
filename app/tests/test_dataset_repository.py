import json
import pytest
from unittest.mock import patch, mock_open, MagicMock
from app.repository.dataset_repository import (
    load_json_from_hf,
    download_pictures_from_hf,
    DatasetRepository
)
import sys

DUMMY_JSON_DATA = {
    "disasters": [
        {
            "name": "hurricane_florence",
            "type": "hurricane",
            "sample_chips": ["chip1", "chip2"]
        },
        {
            "name": "california_wildfires",
            "type": "wildfire",
            "sample_chips": ["chip3"]
        },
        {
            "name": "hurricane_harvey",
            "type": "hurricane",
            "sample_chips": ["chip4"]
        }
    ]
}


@pytest.fixture
def mock_streamlit_secrets():
    with patch("app.repository.dataset_repository.st") as mock_st:
        mock_st.secrets.get.return_value = "fake_hf_token"
        mock_st.cache_data = lambda func: func

        yield mock_st.secrets.get

@pytest.fixture
def mock_streamlit_secrets():
    with patch("app.repository.dataset_repository.st") as mock_get:
        mock_get.return_value = "fake_hf_token"
        yield mock_get


@pytest.fixture
def mock_hf_download():
    with patch("app.repository.dataset_repository.hf_hub_download") as mock_download:
        mock_download.return_value = "/fake/local/path"
        yield mock_download



def test_load_json_from_hf(mock_hf_download, mock_streamlit_secrets):
    m_open = mock_open(read_data=json.dumps(DUMMY_JSON_DATA))

    with patch("builtins.open", m_open):
        load_json_from_hf.clear()
        data = load_json_from_hf()

        assert data == DUMMY_JSON_DATA
        mock_hf_download.assert_called_once()
        mock_streamlit_secrets.secrets.get.assert_called_once_with("HF_TOKEN")
        m_open.assert_called_once_with("/fake/local/path", "r", encoding="utf-8")


def test_download_pictures_from_hf(mock_hf_download, mock_streamlit_secrets):
    sample_pictures = ["chip1"]
    with patch("app.repository.dataset_repository.Image.open") as mock_image_open:
        mock_img_instance = MagicMock()
        mock_image_open.return_value = mock_img_instance

        result = download_pictures_from_hf("hurricane_florence", sample_pictures)

        assert mock_hf_download.call_count == 2
        assert mock_image_open.call_count == 2

        assert "chip1" in result
        assert "pre" in result["chip1"]
        assert "post" in result["chip1"]
        assert result["chip1"]["pre"] == mock_img_instance
        assert result["chip1"]["post"] == mock_img_instance



@pytest.fixture
def repo_instance():
    with patch("app.repository.dataset_repository.load_json_from_hf", return_value=DUMMY_JSON_DATA):
        repo = DatasetRepository()
        return repo


def test_get_json_data(repo_instance):
    assert repo_instance.get_json_data() == DUMMY_JSON_DATA


def test_get_disaster_types(repo_instance):
    types = repo_instance.get_disaster_types()
    assert len(types) == 2
    assert "hurricane" in types
    assert "wildfire" in types


def test_get_disasters_by_type(repo_instance):
    disasters_by_type = repo_instance.get_disasters_by_type()

    assert "hurricane" in disasters_by_type
    assert "wildfire" in disasters_by_type
    assert len(disasters_by_type["hurricane"]) == 2
    assert "hurricane_florence" in disasters_by_type["hurricane"]
    assert "hurricane_harvey" in disasters_by_type["hurricane"]
    assert disasters_by_type["wildfire"] == ["california_wildfires"]


def test_get_disaster_info(repo_instance):
    info = repo_instance.get_disaster_info("hurricane_florence")
    assert info is not None
    assert info["type"] == "hurricane"
    assert info["sample_chips"] == ["chip1", "chip2"]

    missing_info = repo_instance.get_disaster_info("non_existent")
    assert missing_info is None


def test_get_disaster_pictures_cached(repo_instance):
    with patch("app.repository.dataset_repository.download_pictures_from_hf") as mock_download:
        mock_download.return_value = {"chip1": {"pre": "mock_img", "post": "mock_img"}}

        pics1 = repo_instance.get_disaster_pictures("hurricane_florence")
        mock_download.assert_called_once_with("hurricane_florence", ["chip1", "chip2"])

        pics2 = repo_instance.get_disaster_pictures("hurricane_florence")
        assert mock_download.call_count == 1
        assert pics1 == pics2








