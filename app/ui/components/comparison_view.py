import streamlit as st
import itertools
from app.service.inference_service import InferenceService
from app.ui.utils.visualization_functions import merge_mask_with_image

if "inference_service" not in st.session_state:
    st.error("Service not initialized. Please start from the main page.")
    st.stop()


inference_service = st.session_state.inference_service

@st.fragment
def render_comparison_tab(selected_models, model_results, CLASS_LABELS):
    st.header("Deep Dive: Inter-Model Disagreements")
    st.write("Analyze where and how the models predict different damage levels.")

    model_pairs = list(itertools.combinations(selected_models, 2))

    for model_a, model_b in model_pairs:
        res_a = model_results[model_a]
        res_b = model_results[model_b]

        st.subheader(f"{model_a.replace('_', ' ').title()} vs {model_b.replace('_', ' ').title()}")

        view_tabs = st.tabs(["Magnitude Map (Over/Under)", "Targeted Class Isolation"])

        with view_tabs[0]:
            col_text, col_img = st.columns([1, 3])
            with col_text:
                st.markdown(f"**Red:** {model_b} predicted higher damage.")
                st.markdown(f"**Blue:** {model_b} predicted lower damage.")

            with col_img:
                directional_img = inference_service.calculate_directional_disagreement(res_a, res_b)
                combined_img = merge_mask_with_image(
                    base_image=st.session_state.after_img,
                    mask_image=directional_img,
                    target_size=(512, 512)
                )

                st.image(combined_img, use_container_width=False)

        with view_tabs[1]:
            filter_col1, filter_col2 = st.columns(2)

            class_names = [name for idx, name in CLASS_LABELS.items() if idx != 0]

            with filter_col1:
                selected_a_name = st.selectbox(f"{model_a} Predicted:", class_names, key=f"{model_a}_{model_b}_A")
                class_a_idx = list(CLASS_LABELS.keys())[list(CLASS_LABELS.values()).index(selected_a_name)]

            with filter_col2:
                selected_b_name = st.selectbox(f"{model_b} Predicted:", class_names, key=f"{model_a}_{model_b}_B")
                class_b_idx = list(CLASS_LABELS.keys())[list(CLASS_LABELS.values()).index(selected_b_name)]

            target_img, target_rate = inference_service.calculate_targeted_disagreement(
                res_a, res_b, class_a_idx, class_b_idx
            )

            col_metric, col_target_img = st.columns([1, 3])
            with col_metric:
                st.metric(label="Targeted Disagreement", value=f"{target_rate}%")
            with col_target_img:
                combined_img = merge_mask_with_image(
                    base_image=st.session_state.after_img,
                    mask_image=target_img,
                    target_size=(512, 512)
                )

                st.image(combined_img, use_container_width=False)

        st.divider()