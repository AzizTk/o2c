from models.input import EmailInput
from pipeline.decide import make_decision
from pipeline.classify import classify_email

def main():
    # 1. Simulate an incoming email
    email = EmailInput(
        sender="customer@example.com",
        subject="Invoice INV-123 already paid",
        body="Hello, we already paid invoice INV-123 last week. Please check."
    )

    # 2. Classify the email
    case_type, confidence, rationale = classify_email(email)

    # 3. Run decision logic
    decision = make_decision(
        case_type=case_type,
        confidence=confidence,
        invoice_ids=["INV-123"],
        amount=None,
        rationale=rationale,
    )

    # 4. Output result
    print("Email received:")
    print(email.model_dump())
    print("\nFinal decision:")
    print(decision.model_dump())


if __name__ == "__main__":
    main()
