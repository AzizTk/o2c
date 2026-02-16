from models.decision import Decision

CONFIDENCE_THRESHOLD = 0.8


def make_decision(
    case_type: str,
    confidence: float,
    invoice_ids: list[str] | None,
    amount: float | None,
    rationale: str,
) -> Decision:
    """
    Simple decision logic based on confidence and case type.
     - If confidence is high and case type is known, auto-reply.
     - If confidence is low or case type is unknown, queue for review.
     - If no invoice IDs are found, queue for review regardless of confidence.
     - Rationale is included for transparency in the decision-making process.
    """
    action = "QUEUE_FOR_REVIEW"
    if confidence >= CONFIDENCE_THRESHOLD and case_type != "UNKNOWN":
        action = "AUTO_REPLY"
    if not invoice_ids:
        action = "QUEUE_FOR_REVIEW"
    return Decision(
        case_type=case_type,
        invoice_ids=invoice_ids,
        amount=amount,
        confidence=confidence,
        action=action,
        rationale=rationale,
    )
