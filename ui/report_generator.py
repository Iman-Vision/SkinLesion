"""
DERMA.AI Clinical Report Generator
Generates a PDF analysis report for research and clinical consultation documentation.
Sanitizes all text inputs against XML/HTML parsing errors in ReportLab.
"""

import io
import html
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image


def _xml_escape(text: str) -> str:
    """Escapes strings for ReportLab Paragraph XML parsing (&, <, >)."""
    if not isinstance(text, str):
        text = str(text)
    return html.escape(text.strip())


def generate_pdf_report(
    analysis_id: str,
    timestamp: str,
    filename: str,
    image: Image.Image,
    top_code: str,
    top_name: str,
    top_conf: float,
    is_high_risk: bool,
    class_probs: list,
    class_names: list,
    full_names: dict,
    model_name: str = "MobileNetV2 (HAM10000)"
) -> bytes:
    """
    Builds a professional clinical AI assessment report PDF in memory and returns raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Brand palette
    c_primary = colors.HexColor('#172321')
    c_secondary = colors.HexColor('#263832')
    c_accent = colors.HexColor('#477568')
    c_light_bg = colors.HexColor('#F8F9F7')
    c_border = colors.HexColor('#E2E8E3')
    c_muted = colors.HexColor('#6D7972')

    # Custom typography styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary
    )
    style_heading = ParagraphStyle(
        'DocHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_secondary
    )
    style_body = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_primary
    )
    style_meta = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_muted
    )

    story = []

    # Sanitize user and dynamic values
    safe_aid = _xml_escape(analysis_id)
    safe_time = _xml_escape(timestamp)
    safe_model = _xml_escape(model_name)
    safe_filename = _xml_escape(filename)
    safe_top_name = _xml_escape(top_name)
    safe_top_code = _xml_escape(top_code).upper()

    # 1. Header with branding and report metadata
    header_data = [
        [
            Paragraph("<b>DERMA.AI</b><br/><font color='#477568' size=8>Intelligent Skin Analysis System</font>", style_title),
            Paragraph(f"<b>Report ID:</b> {safe_aid}<br/><b>Generated:</b> {safe_time}<br/><b>Model:</b> {safe_model}", style_meta)
        ]
    ]
    header_table = Table(header_data, colWidths=[300, 232])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 1, c_accent),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    # 2. Main content: Image thumbnail + Primary Assessment Card
    thumb_img = image.copy()
    thumb_img.thumbnail((180, 180))
    img_buf = io.BytesIO()
    thumb_img.save(img_buf, format='JPEG', quality=90)
    img_buf.seek(0)
    rl_img = RLImage(img_buf, width=150, height=150)

    risk_text = "<font color='#8B3A3A'><b>Higher Clinical Concern</b></font>" if is_high_risk else "<font color='#3D6B4F'><b>Typically Lower Concern</b></font>"

    assessment_html = f"""
    <b>Primary Model Classification:</b><br/>
    <font size=14 color='#172321'><b>{safe_top_name}</b></font> ({safe_top_code})<br/><br/>
    <b>Classification Confidence:</b> <font size=12><b>{top_conf*100:.1f}%</b></font><br/>
    <b>Clinical Category Tier:</b> {risk_text}<br/>
    <b>Source Filename:</b> {safe_filename}<br/>
    <b>Input Dimensions:</b> {image.size[0]} × {image.size[1]} px
    """

    overview_data = [
        [rl_img, Paragraph(assessment_html, style_body)]
    ]
    overview_table = Table(overview_data, colWidths=[170, 362])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 16))

    # 3. Probability Distribution Table
    story.append(Paragraph("<b>Complete 7-Class Probability Distribution</b>", style_heading))
    story.append(Spacer(1, 6))

    prob_rows = [
        [Paragraph("<b>Class Code</b>", style_body), Paragraph("<b>Full Diagnosis Name</b>", style_body), Paragraph("<b>Probability</b>", style_body), Paragraph("<b>Clinical Tier</b>", style_body)]
    ]

    # Sort descending
    sorted_indices = sorted(range(len(class_probs)), key=lambda i: class_probs[i], reverse=True)
    for idx in sorted_indices:
        code = class_names[idx]
        name = _xml_escape(full_names.get(code, code))
        pct = f"{class_probs[idx] * 100:.2f}%"
        is_hr = code in {'akiec', 'bcc', 'mel'}
        tier = "Higher Concern" if is_hr else "Usually Benign"
        prob_rows.append([
            Paragraph(f"<b>{code.upper()}</b>", style_body),
            Paragraph(name, style_body),
            Paragraph(pct, style_body),
            Paragraph(tier, style_body)
        ])

    prob_table = Table(prob_rows, colWidths=[80, 230, 100, 122])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_light_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), c_secondary),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 16))

    # 4. Mandatory Clinical Disclaimer
    disclaimer_html = """
    <b>IMPORTANT MEDICAL DISCLAIMER:</b><br/>
    DERMA.AI is an assistive artificial intelligence technology developed strictly for research, clinical education,
    and decision-support experimentation. It is NOT an FDA-cleared diagnostic device and does not establish a definitive
    medical diagnosis. Artificial intelligence predictions are statistical pattern recognitions that must never supersede
    in-person examination, dermoscopy, or histopathological biopsy by a licensed dermatologist.
    """
    disclaimer_table = Table([[Paragraph(disclaimer_html, style_meta)]], colWidths=[532])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_accent),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(KeepTogether(disclaimer_table))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
