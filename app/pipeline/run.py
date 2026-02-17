from typing import List
from models.input import EmailInput
from pipeline.classify import classify_email
from pipeline.decide import make_decision
from models.decision import Decision


def process_emails(emails: List[EmailInput]) -> List[Decision]:
    decisions = []

    for email in emails:
        case_type, confidence, rationale = classify_email(email)

        decision = make_decision(
            case_type=case_type,
            confidence=confidence,
            invoice_ids=["INV-123"],  # placeholder
            amount=None,
            rationale=rationale,
        )

        decisions.append(decision)

    return decisions
