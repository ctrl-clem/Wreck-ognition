import streamlit as st
from app.service.report_service import ReportService



@st.fragment
def download_section():
    st.button("Generate Final PDF", type="primary", key='generate_button')
    if st.session_state.get("generate_button") == True:
            with st.spinner("Compiling everything into a PDF..."):
                st.session_state.pdf_bytes = st.session_state.report_service.generate_pdf(st.session_state.report_artifacts)

            st.download_button(
                    label="Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="Damage_Assessment_Report.pdf",
                    mime="application/pdf"
            )

