from models.input import EmailInput


def classify_email(email: EmailInput) -> tuple[str, float, str]:
    """
    Mock AI classifier.
    Returns: (case_type, confidence, rationale)
    """

    text = f"{email.subject} {email.body}".lower()

    if "deduct" in text or "deduction" in text:
        return "DEDUCTION", 0.9, "Email mentions a deduction"

    if "paid" in text or "payment" in text:
        return "PAYMENT_ISSUE", 0.9, "Email mentions payment already made"

    if "dispute" in text or "wrong" in text or "incorrect" in text:
        return "DISPUTE", 0.85, "Email disputes the charge"

    return "UNKNOWN", 0.6, "Unable to clearly classify the email"
