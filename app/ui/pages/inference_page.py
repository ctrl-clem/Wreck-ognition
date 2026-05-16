import streamlit as st
from PIL import Image
import torch
import os
from app.ui.utils.visualization_functions import display_overlay, display_distribution_chart, merge_mask_with_image, generate_cam_visualization
from app.config import CLASS_COLORS, CLASS_LABELS
from app.ui.utils.default_data import get_default_data
import pandas as pd
from app.repository.schema import ModelInput, PredictionResult
from streamlit_image_comparison import image_comparison
from app.ui.components.comparison_view import render_comparison_tab
from app.ui.components.inference_tabs import render_inference_tabs
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import itertools
from app.domain.report.report_schema import FullReportData
from app.service.report_service import ReportService
from app.ui.components.download_section import download_section



if "inference_service" not in st.session_state:
    st.error("Service not initialized. Please start from the main page.")
    st.stop()


if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

if "before_img" not in st.session_state:
    st.session_state.before_img = None
if "after_img" not in st.session_state:
    st.session_state.after_img = None
if "mask_img" not in st.session_state:
    st.session_state.mask_img = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "report_artifacts" not in st.session_state:
    st.session_state.report_artifacts = FullReportData(
        project_id="Ablation Study",
        pre_image=st.session_state.before_img,
        post_image=st.session_state.after_img,
        models={}
    )

inference_service = st.session_state.inference_service
report_service = st.session_state.report_service

PAGE_ID = "inference"

st.title("Disaster Damage Ablation Study")
st.markdown("---")

st.subheader("1. Upload Imagery")

if st.button("Use Default", type="primary"):
    before_file, after_file, mask_file = get_default_data()
    st.session_state.before_img = before_file
    st.session_state.after_img = after_file
    st.session_state.mask_img = mask_file

    st.session_state.uploader_key += 1

col1, col2, col3 = st.columns(3)

with col1:
    before_file = st.file_uploader(
        "Upload 'Before' Image",
        type=['png', 'jpg', 'jpeg'],
        key=f"{PAGE_ID}_before_upload_{st.session_state.uploader_key}"
    )
    if before_file is not None:
        st.session_state.before_img = before_file

    if st.session_state.before_img is not None:
        st.image(st.session_state.before_img, caption="Pre-Disaster", width='content')

with col2:
    after_file = st.file_uploader(
        "Upload 'After' Image",
        type=['png', 'jpg', 'jpeg'],
        key=f"{PAGE_ID}_after_upload_{st.session_state.uploader_key}"
    )
    if after_file is not None:
        st.session_state.after_img = after_file

    if st.session_state.after_img is not None:
        st.image(st.session_state.after_img, caption="Post-Disaster", width='content')

with col3:
    mask_file = st.file_uploader(
        "Upload Pre-disaster Mask",
        type=['png', 'jpg', 'jpeg'],
        key=f"{PAGE_ID}_mask_upload_{st.session_state.uploader_key}"
    )
    if mask_file is not None:
        st.session_state.mask_img = mask_file

    if st.session_state.mask_img is not None:
        st.image(st.session_state.mask_img, caption="Building Mask", width='content')

st.markdown("---")

st.subheader("2. Model Configuration")
st.write("Check the boxes of the models that you want to run:")

model_options = {
    "post_only": st.checkbox("post-only"),
    "pre_post": st.checkbox("pre-post"),
    "pre_post_premask": st.checkbox("pre-post-premask"),
    "post_premask": st.checkbox("post-premask")
}

if st.button("Run Analysis", type="primary", width='content'):
    st.session_state.analysis_complete = True
    selected_models = [name for name, checked in model_options.items() if checked]
    errors = []

    if not selected_models:
        st.warning("Please select at least one model to run.")
    else:
        if "post_only" in selected_models and not st.session_state.after_img:
            errors.append("Model **post-only** requires the 'After' image.")
        if "pre_post" in selected_models and not (st.session_state.before_img and st.session_state.after_img):
            errors.append("Model **pre-post** requires both 'Before' and 'After' images.")
        if "pre_post_premask" in selected_models and not (st.session_state.before_img and st.session_state.after_img and st.session_state.mask_img):
            errors.append("Model **pre-post-premask** requires ALL three inputs.")
        if "post_premask" in selected_models and not (st.session_state.after_img and st.session_state.mask_img):
            errors.append("Model **post-premask** requires 'After' and 'Mask' images.")

    if errors:
        for error in errors:
            st.error(error)
        st.session_state.analysis_complete = False
    else:
        st.session_state.selected_models = selected_models
        st.session_state.pdf_bytes = None
        st.session_state.analysis_complete = True

    if st.session_state.get("analysis_complete") == True:
        tabs, results = render_inference_tabs(
            selected_models=st.session_state.selected_models,
            inference_service=inference_service,
            pre_img=st.session_state.before_img,
            post_img=st.session_state.after_img,
            premask=st.session_state.mask_img,
            class_labels=CLASS_LABELS,
            class_colors=CLASS_COLORS,
        )

        if len(st.session_state.selected_models) > 1:
            with tabs[-1]:
                render_comparison_tab(
                    selected_models=st.session_state.selected_models,
                    model_results=results,
                    CLASS_LABELS=CLASS_LABELS,
                )

        download_section()








    #
    # st.write("AICI")
    # st.write(st.session_state.report_artifacts.models)
    # st.write(st.session_state.report_artifacts.post_image)
    # st.write(st.session_state.report_artifacts.pre_image)
    # st.write(st.session_state.report_artifacts.comparison)
    # st.write(st.session_state.report_artifacts.project_id)
    # st.write(st.session_state.report_artifacts.common_legends)


