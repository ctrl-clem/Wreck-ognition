import streamlit as st
import pandas as pd
from app.ui.components.slider import mask_slider
from app.ui.utils.visualization_functions import ( display_distribution_chart,
                                                   display_overlay,
                                                   merge_mask_with_image,
                                                   generate_cam_visualization,
                                                   to_heatmap,
                                                   get_heatmap_legend,
                                                   merge_high_contrast_heatmap,
                                                   fig_to_pil
                                                   )
from app.domain.report.report_schema import FullReportData, EntropyArtifact, ModelArtifacts, ComparisonArtifacts

@st.fragment
def render_inference_tabs(
        selected_models,
        inference_service,
        pre_img,
        post_img,
        premask,
        class_labels,
        class_colors
):
    tab_titles = [m.replace('_', ' ').title() for m in selected_models]
    if len(tab_titles) > 1:
        tab_titles.append("Comparison")

    tabs = st.tabs(tab_titles)
    results = {}

    for i, model_name in enumerate(selected_models):
        with tabs[i]:
            result = inference_service.run_inference(
                model_type=model_name,
                pre_img=pre_img,
                post_img=post_img,
                premask=premask
            )
            results[model_name] = result

            model_entry = ModelArtifacts(
                model_name=model_name,
                inference_time=result.latency,
                damage_density=result.damage_density,
                prediction_overlay=result.colored_overlay,
                confidence_levels=result.confidence_scores,
                gradcam_image=None,
                entropies=[],
                distribution_graph=None
            )

            # Header Section
            st.title(f" {model_name.replace('_', ' ').title()} Analysis")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Damage Density", f"{result.damage_density}%", help="Percentage of buildings damaged.")
            with m_col2:
                st.metric("Latency", f"{result.latency}s", help="Model inference time.")

            st.markdown("---")

            # Main Display: Prediction and Statistics
            col_img, col_stats = st.columns([1.2, 0.8])

            with col_img:
                st.subheader("Prediction Overlay")
                mask_slider(
                    post_image=post_img,
                    mask_overlay=result.colored_overlay,
                    height=512,
                    key=f"slider_{model_name}_{i}"
                )

                with st.expander("Color Legend", expanded=False):
                    cols = st.columns(2)
                    for idx, name in class_labels.items():
                        if name != 'Background':
                            color = class_colors[idx]
                            target_col = cols[0] if idx < 3 else cols[1]
                            target_col.markdown(
                                f'<p style="margin:0;"><span style="color:rgb({int(color[0])},{int(color[1])},{int(color[2])});">■</span> {name}</p>',
                                unsafe_allow_html=True
                            )

            with col_stats:
                st.subheader("Classification Analytics")
                distribution_graph = display_distribution_chart(result.class_distribution, class_colors, class_labels)
                st.pyplot(distribution_graph)

                model_entry.distribution_graph = distribution_graph

                conf_df = pd.DataFrame([
                    {"Class": label, "Confidence": f"{score}%"}
                    for label, score in result.confidence_scores.items() if score > 0
                ])
                st.dataframe(conf_df, hide_index=True, use_container_width=True)

            st.markdown("---")

            # GradCAM Section
            st.subheader("Grad-CAM Explainability")
            cam_img = generate_cam_visualization(post_img, result.gradcam_post)
            st.image(cam_img, width=512, caption="Saliency map showing model focus areas.")
            model_entry.gradcam_image = cam_img
            if premask is not None:
                st.markdown("---")
                st.subheader("Spatial Uncertainty Audits")

                premask_np = inference_service.convert_premask_to_np(premask)
                ent_col1, ent_col2, ent_col3 = st.columns(3)
                entropy_list = []
                # Column 1: Agreement
                with ent_col1:
                    st.markdown("**Agreement Audit**")
                    agree_map, agree_score = inference_service.compute_agreement_entropy(result.logits, premask_np)
                    # Use use_container_width=False to enforce the 512 limit if the column is wider
                    viz = merge_high_contrast_heatmap(post_img, to_heatmap(agree_map))
                    st.image(viz, width=512,
                             use_container_width=False)
                    st.metric("Agreement Entropy", agree_score)
                    entropy_list.append((EntropyArtifact(name="Agreement",score=agree_score,image=viz)))

                # Column 2: Hallucination
                with ent_col2:
                    st.markdown("**Hallucination Audit**")
                    hallu_map, hallu_score = inference_service.compute_hallucination_entropy(result.logits, premask_np)
                    viz = merge_high_contrast_heatmap(post_img, to_heatmap(hallu_map))
                    st.image(viz, width=512,
                             use_container_width=False)
                    st.metric("Hallucination Entropy", hallu_score)
                    entropy_list.append((EntropyArtifact(name="Hallucination",score=hallu_score,image=viz)))


                # Column 3: Omission
                with ent_col3:
                    st.markdown("**Omission Audit**")
                    omis_map, omis_score = inference_service.compute_omission_entropy(result.logits, premask_np)
                    viz = merge_high_contrast_heatmap(post_img, to_heatmap(omis_map))
                    st.image(viz, width=512,
                             use_container_width=False)
                    st.metric("Omission Entropy", omis_score)
                    entropy_list.append((EntropyArtifact(name="Omission", score=omis_score, image=viz)))

                # Legend Centered below the images
                _, leg_col, _ = st.columns([1, 2, 1])
                with leg_col:
                    st.image(get_heatmap_legend(), use_container_width=True)

                model_entry.entropies = entropy_list

        st.session_state.report_artifacts.models[model_name] = model_entry


    return tabs, results