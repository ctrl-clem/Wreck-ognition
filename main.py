import streamlit as st
from app.domain.models.model_factory import ModelFactory
from app.service.inference_service import InferenceService
from app.service.report_service import ReportService
from app.repository.dataset_repository import DatasetRepository
from app.service.dataset_service import DatasetService

st.set_page_config(page_title="Disaster Analysis Thesis", layout="wide")
if "inference_service" not in st.session_state:
    st.session_state.inference_service = InferenceService()


if "report_service" not in st.session_state:
    st.session_state.report_service = ReportService()

if "dataset_repository" not in st.session_state:
    st.session_state.dataset_repository = DatasetRepository()

if "dataset_service" not in st.session_state:
    st.session_state.dataset_service = DatasetService(st.session_state.dataset_repository)



inference_service = st.session_state.inference_service
report_serivce = st.session_state.report_service
dataset_service = st.session_state.dataset_service

inference_page = st.Page("app/ui/pages/inference_page.py", title="Inference & Ablation", default=True)
dataset_statistics_page = st.Page("app/ui/pages/dataset_statistics_page.py", title="Dataset statistics")
edit_studio_page = st.Page("app/ui/pages/edit_studio_page.py", title="Editing Studio")

pg = st.navigation([inference_page,dataset_statistics_page, edit_studio_page])

pg.run()
# weights = WEIGHTS_FOLDER + ""
#
# # The factory ensures this is only "heavy" the first time
# model = ModelFactory.get_model(model_choice, weights)
#
# if st.button("Run Classification"):
#     result = model.predict(user_inputs)
#     st.image(result.colored_overlay)
