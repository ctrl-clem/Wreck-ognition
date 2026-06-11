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

pg = st.navigation({
    "Core Features": [inference_page],
    "Data & Tools": [dataset_statistics_page, edit_studio_page]
})

st.markdown("""
    <style>
        /* ── Sidebar base ── */
        [data-testid="stSidebar"] {
            background-color: #0f1117;
            border-right: 1px solid #1e2130;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        /* ── App title / logo area ── */
        [data-testid="stSidebarHeader"] {
            padding: 0 1.25rem 1rem;
            border-bottom: 1px solid #1e2130;
            margin-bottom: 0.5rem;
        }

        /* ── Section group labels ── */
        [data-testid="stSidebarNav"] > ul > li > span,
        [data-testid="stSidebarNav"] .st-emotion-cache-1cypcdb {
            font-size: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: #4a5568 !important;
            padding: 1.25rem 1rem 0.4rem !important;
            display: block;
        }

        /* ── Nav links ── */
        [data-testid="stSidebarNav"] ul {
            padding: 0 0.75rem;
            gap: 2px;
        }

        [data-testid="stSidebarNav"] ul li a {
            display: flex !important;
            align-items: center !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #8892a4 !important;
            border-radius: 6px !important;
            padding: 0.55rem 0.85rem !important;
            transition: background 0.15s ease, color 0.15s ease !important;
            text-decoration: none !important;
            letter-spacing: 0.01em;
        }

        /* ── Hover state ── */
        [data-testid="stSidebarNav"] ul li a:hover {
            background-color: #1a1f2e !important;
            color: #e2e8f0 !important;
        }

        /* ── Active / selected page ── */
        [data-testid="stSidebarNav"] ul li a[aria-current="page"] {
            background-color: #1a2744 !important;
            color: #6ea8fe !important;
            font-weight: 600 !important;
            border-left: 3px solid #6ea8fe;
            padding-left: calc(0.85rem - 3px) !important;
        }

        /* ── Active hover (keep it consistent) ── */
        [data-testid="stSidebarNav"] ul li a[aria-current="page"]:hover {
            background-color: #1e2d52 !important;
        }

        /* ── Divider between section groups ── */
        [data-testid="stSidebarNav"] > ul > li + li > span {
            border-top: 1px solid #1e2130;
            margin-top: 0.5rem;
        }

        /* ── Scrollbar ── */
        [data-testid="stSidebar"] ::-webkit-scrollbar {
            width: 4px;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar-track {
            background: transparent;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
            background: #2d3448;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)

pg.run()
# weights = WEIGHTS_FOLDER + ""
#
# model = ModelFactory.get_model(model_choice, weights)
#
# if st.button("Run Classification"):
#     result = model.predict(user_inputs)
#     st.image(result.colored_overlay)
