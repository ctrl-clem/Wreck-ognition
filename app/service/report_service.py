import io
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from app.domain.report.report_schema import FullReportData

class ReportService:
    def generate_pdf(self, report_data: FullReportData) -> bytes:
        # Create an in-memory buffer for the final PDF
        pdf_buffer = io.BytesIO()

        # Initialize the Document Template
        doc = SimpleDocTemplate(
            pdf_buffer, pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=54, bottomMargin=54
        )

        story = []  # This is the sequence of elements that will be drawn to the PDF
        styles = getSampleStyleSheet()

        # --- Custom Typographic Styles ---
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, alignment=TA_CENTER,
                                     spaceAfter=20)
        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER)
        section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=14, spaceBefore=20,
                                       spaceAfter=10, backColor=colors.lightgrey)
        metric_style = ParagraphStyle('MetricStyle', parent=styles['Normal'], fontSize=11, spaceAfter=4)
        styles.add(ParagraphStyle('Caption', parent=styles['Italic'], alignment=TA_CENTER, fontSize=9))

        # --- COVER PAGE ---
        story.append(Spacer(1, 2 * inch))
        story.append(Paragraph("Model Comparison Audit", title_style))
        story.append(Paragraph(f"Project: {report_data.project_id}", subtitle_style))
        story.append(PageBreak())

        # --- INPUT CONTEXT PAGE ---
        story.append(Paragraph("Input Imagery Context", section_style))
        story.append(self.create_dual_image_table(
            report_data.pre_image, "Pre-Disaster Satellite",
            report_data.post_image, "Post-Disaster Satellite",
            styles
        ))

        # --- INDIVIDUAL MODEL ANALYSIS LOOP ---
        for model_name, artifact in report_data.models.items():
            story.append(PageBreak())
            story.append(Paragraph(f"Model: {model_name.upper()}", section_style))

            # Metrics
            story.append(Paragraph(f"<b>Inference Latency:</b> {artifact.inference_time} seconds", metric_style))
            story.append(Paragraph(f"<b>Damage Density:</b> {artifact.damage_density}%", metric_style))
            story.append(Spacer(1, 15))

            # Row 1: Overlay and Distribution Chart
            story.append(self.create_dual_image_table(
                artifact.prediction_overlay, "Prediction Mask Overlay",
                artifact.distribution_graph, "Class Distribution Analysis",
                styles
            ))
            story.append(Spacer(1, 20))

            # Row 2: Explainability (Grad-CAM)
            story.append(Paragraph("<b>Explainability & Attention (Grad-CAM)</b>", metric_style))
            story.append(Spacer(1, 5))
            # Just one image, so we add it directly
            story.append(self.process_image_to_flowable(artifact.gradcam_image, width=3.5 * inch, height=3.5 * inch))
            story.append(Spacer(1, 20))

            # Row 3: Entropy Audits
            story.append(Paragraph("<b>Spatial Uncertainty Audits (Entropy)</b>", metric_style))
            story.append(Spacer(1, 10))

            for i in range(0, len(artifact.entropies), 2):
                batch = artifact.entropies[i:i + 2]
                img1 = batch[0].image
                cap1 = f"{batch[0].name} (Score: {batch[0].score})"

                img2 = batch[1].image if len(batch) > 1 else None
                cap2 = f"{batch[1].name} (Score: {batch[1].score})" if len(batch) > 1 else ""

                story.append(self.create_dual_image_table(img1, cap1, img2, cap2, styles))
                story.append(Spacer(1, 10))

        # Build the PDF
        doc.build(story, onFirstPage=self.add_header_footer, onLaterPages=self.add_header_footer)

        # Return the pure binary string needed by Streamlit
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    def add_header_footer(self,canvas, doc):
        """Draws the header and footer on every page."""
        canvas.saveState()
        # Header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        canvas.drawCentredString(letter[0] / 2.0, letter[1] - 0.5 * inch,
                                 "Ablation Study: Disaster Damage Assessment Report")
        # Footer
        canvas.setFont('Helvetica-Oblique', 8)
        canvas.drawCentredString(letter[0] / 2.0, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    def process_image_to_flowable(self,img_data, width=3.2 * inch, height=3.2 * inch):
        """Safely converts Matplotlib, Numpy, or PIL into an in-memory ReportLab Image."""
        if img_data is None:
            return Paragraph("No Image Available")

        # 1. Convert everything to PIL Image first
        if hasattr(img_data, 'savefig'):  # Matplotlib Figure
            buf = io.BytesIO()
            img_data.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
        elif isinstance(img_data, np.ndarray):  # GradCAM Array
            img = Image.fromarray(img_data)
        else:  # Already a PIL Image
            img = img_data

        # 2. Ensure RGB mode (prevents transparency PDF crashes)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 3. Save to an in-memory byte buffer instead of a physical file
        img_buf = io.BytesIO()
        img.save(img_buf, format='PNG')
        img_buf.seek(0)

        # 4. Return as a ReportLab flowable component
        return RLImage(img_buf, width=width, height=height)

    def create_dual_image_table(self, img1_data, cap1, img2_data, cap2, styles):
        """Creates a side-by-side layout for two images and their captions."""
        img1 = self.process_image_to_flowable(img1_data)
        img2 = self.process_image_to_flowable(img2_data) if img2_data is not None else Paragraph("")

        p_cap1 = Paragraph(cap1, styles['Caption'])
        p_cap2 = Paragraph(cap2, styles['Caption']) if cap2 else Paragraph("")

        data = [
            [img1, img2],
            [p_cap1, p_cap2]
        ]

        # Build the layout table
        t = Table(data, colWidths=[3.5 * inch, 3.5 * inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 1), (-1, 1), 6),  # Space between image and caption
        ]))
        return t