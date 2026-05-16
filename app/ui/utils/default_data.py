import streamlit as st
from app.config import DEFAULT_DATA
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

def get_default_data():
    try:
        before_image = DEFAULT_DATA["pre"]
        after_image = DEFAULT_DATA["post"]
        premask = DEFAULT_DATA["premask"]

        return before_image, after_image, premask
    except (UnidentifiedImageError, ValueError, TypeError) as e:
        raise ValueError(f"The uploaded file is not a valid image. Details: {e}")

