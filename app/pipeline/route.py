from pipeline.classify import ClassificationResult


def route_email(classification: ClassificationResult) -> str:
    """
    Routes an email to the correct queue based on classification.

    args:
        classification: ClassificationResult object containing case_type and confidence
    
    returns:
        queue name as string
    """

    if classification.case_type == "Payment Claim":
        return "Cash Application"

    if classification.case_type == "Dispute":
        return "Disputes"

    # UNKNOWN or anything else
    return "AR Support"
