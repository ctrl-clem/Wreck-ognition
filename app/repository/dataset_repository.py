import io
import json
import streamlit as st
from huggingface_hub import hf_hub_download
from app.config import HF_DATASET_ID, JSON_FILENAME
from PIL import Image


@st.cache_data
def load_json_from_hf() -> dict | list:
    local_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=JSON_FILENAME,
        repo_type="dataset",
        token=st.secrets.get("HF_TOKEN"),
    )
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_pictures_from_hf(disaster_name, sample_pictures):
    images = {}
    for picture in sample_pictures:
        pre_filename = f"Samples/{disaster_name}/{picture}_pre_disaster.png"
        post_filename = f"Samples/{disaster_name}/{picture}_post_disaster.png"

        pre_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=pre_filename,
            repo_type="dataset",
            token=st.secrets.get("HF_TOKEN")
        )

        pre_img = Image.open(pre_path)

        post_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=post_filename,
            repo_type="dataset",
            token=st.secrets.get("HF_TOKEN")
        )

        post_img = Image.open(post_path)

        images[picture] = {
            "pre":pre_img,
            "post":post_img
        }
    return images




class DatasetRepository:
    def __init__(self):
        self._json_data: dict | list = load_json_from_hf()
        self._sample_pictures = {}

    def get_json_data(self) -> dict | list:
        return self._json_data

    def get_disaster_types(self):
        names = set()
        for disaster in self._json_data["disasters"]:
            names.add(disaster["type"])

        return list(names)

    def get_disasters_by_type(self):
        disasters = {}

        for disaster in self._json_data["disasters"]:
            d_type = disaster["type"]
            d_name = disaster["name"]
            if d_type not in disasters:
                disasters[d_type] = []
            disasters[d_type].append(d_name)

        return disasters

    def get_disaster_info(self, disaster_name):
        disaster_info = None
        for disaster in self._json_data["disasters"]:
            d_name = disaster["name"]
            if d_name == disaster_name:
                disaster_info = disaster
                break

        return disaster_info

    def get_disaster_pictures(self,disaster_name):
        if disaster_name not in self._sample_pictures:
            disaster_info = self.get_disaster_info(disaster_name)
            if not disaster_info:
                return {}

            downloaded_data = download_pictures_from_hf(disaster_name,disaster_info["sample_chips"])
            self._sample_pictures[disaster_name] = downloaded_data

        return self._sample_pictures[disaster_name]



