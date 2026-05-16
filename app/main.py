import streamlit as st
from app.domain.models.model_factory import ModelFactory
from app.service.inference_service import InferenceService
from app.service.report_service import ReportService

st.set_page_config(page_title="Disaster Analysis Thesis", layout="wide")
if "inference_service" not in st.session_state:
    st.session_state.inference_service = InferenceService()


if "report_service" not in st.session_state:
    st.session_state.report_service = ReportService()



inference_service = st.session_state.inference_service
report_serivce = st.session_state.report_service

inference_page = st.Page("ui/pages/inference_page.py", title="Inference & Ablation", default=True)

pg = st.navigation([inference_page])

pg.run()
# weights = WEIGHTS_FOLDER + ""
#
# # The factory ensures this is only "heavy" the first time
# model = ModelFactory.get_model(model_choice, weights)
#
# if st.button("Run Classification"):
#     result = model.predict(user_inputs)
#     st.image(result.colored_overlay)
