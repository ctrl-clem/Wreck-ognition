import streamlit as st
from app.service.dataset_service import DatasetService
from app.ui.components.statistics_page_components import render_disaster_groups,convert_image_to_bytes
from app.ui.utils.visualization_functions import display_distribution_chart_statistics_page, display_building_pie_chart

dataset_service = st.session_state.dataset_service

disasters_by_type = dataset_service.get_disasters_by_type()

if "selected_disaster" not in st.session_state:
    st.session_state.selected_disaster = None

st.session_state.selected_disaster = render_disaster_groups(disasters_by_type)

if st.session_state.selected_disaster:
    st.divider()
    with st.spinner("Loading..."):
        disaster_info = dataset_service.get_disaster_info(st.session_state.selected_disaster)
        pictures = dataset_service.get_disaster_pictures(st.session_state.selected_disaster)


    if disaster_info:
        pretty_title = st.session_state.selected_disaster.replace("-", " ").title()
        st.markdown(f"### {pretty_title} Overview")

        metric_col1, metric_col2, metric_col3, _ = st.columns([1, 1, 1, 1])

        with metric_col1:
            st.metric(label="Total Images", value=disaster_info.get("num_images", 0))

        with metric_col2:
            avg_bldgs = disaster_info.get("avg_buildings_per_image", 0)
            st.metric(label="Avg Buildings / Image",
                      value=f"{avg_bldgs:.1f}" if isinstance(avg_bldgs, (float, int)) else avg_bldgs)

        st.write("")
        st.write("")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            bar_fig = display_distribution_chart_statistics_page(disaster_info)
            if bar_fig:
                st.pyplot(bar_fig, use_container_width=True)

        with chart_col2:
            pie_fig = display_building_pie_chart(disaster_info)
            if pie_fig:
                st.pyplot(pie_fig, use_container_width=True)

        for name, images in pictures.items():
            st.write(f"### {name}")
            col1, col2 = st.columns(2)
            with col1:
                if images["pre"]:
                    st.image(images["pre"], caption="Pre-Disaster")

                    img_bytes = convert_image_to_bytes(images["pre"])
                    st.download_button(
                        label="Download Picture",
                        data=img_bytes,
                        file_name=f"{name}_pre_disaster.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_pre_{name}"
                    )
            with col2:
                if images["post"]:
                    st.image(images["post"], caption="Post-Disaster")

                    img_bytes = convert_image_to_bytes(images["post"])
                    st.download_button(
                        label="Download Picture",
                        data=img_bytes,
                        file_name=f"{name}_post_disaster.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_post_{name}"
                    )