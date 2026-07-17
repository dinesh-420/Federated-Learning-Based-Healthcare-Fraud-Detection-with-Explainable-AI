from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def generate_pdf(
    filename,
    result,
    confidence,
    fraud_probability,
    model_name,
    shap_explanations
):
    styles = getSampleStyleSheet()

    pdf = SimpleDocTemplate(filename)

    elements = []

    # Remove emojis from prediction
    clean_result = (
        result.replace("✅ ", "")
              .replace("🚨 ", "")
    )

    elements.append(
        Paragraph("<b>Healthcare Fraud Detection Report</b>", styles["Title"])
    )

    elements.append(
        Paragraph(f"<b>Prediction:</b> {clean_result}", styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Confidence:</b> {confidence}%", styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Fraud Probability:</b> {fraud_probability}%", styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Model:</b> {model_name}", styles["Normal"])
    )

    elements.append(
        Paragraph("<br/><b>Top Factors Influencing Prediction</b>", styles["Heading2"])
    )

    for feature, impact, explanation in shap_explanations:

        feature_name = feature.replace("_", " ").title()

        elements.append(
            Paragraph(
                f"<b>{feature_name}</b>: {impact} ({explanation})",
                styles["Normal"]
            )
        )

    pdf.build(elements)