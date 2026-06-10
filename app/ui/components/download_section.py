import streamlit as st
from app.service.report_service import ReportService



@st.fragment
def download_section():
    st.button("Generate Final PDF", type="primary", key='generate_button')
    if st.session_state.get("generate_button") == True:
            with st.spinner("Compiling everything into a PDF..."):
                st.session_state.report_artifacts.pre_image = st.session_state.before_img
                st.session_state.report_artifacts.post_image = st.session_state.after_img

                st.write(st.session_state.report_artifacts.models)
                st.write(st.session_state.report_artifacts.post_image)
                st.write(st.session_state.report_artifacts.pre_image)
                st.write(st.session_state.report_artifacts.comparison)
                st.write(st.session_state.report_artifacts.project_id)
                st.write(st.session_state.report_artifacts.common_legends)

                st.session_state.pdf_bytes = st.session_state.report_service.generate_pdf(st.session_state.report_artifacts)

            st.download_button(
                    label="Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="Damage_Assessment_Report.pdf",
                    mime="application/pdf"
            )

