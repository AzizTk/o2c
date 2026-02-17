from pipeline.classify import ClassificationResult


def route_email(classification: ClassificationResult) -> str:
    """
    Routes an email to the correct queue based on classification.
    """

    if classification.case_type == "PAYMENT_ISSUE":
        return "Cash Application"

    if classification.case_type in {"DEDUCTION", "DISPUTE"}:
        return "Disputes"

    # UNKNOWN or anything else
    return "AR Support"
