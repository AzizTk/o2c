from models.input import EmailInput
from pipeline.decide import make_decision


def main():
    # 1. Simulate an incoming email
    email = EmailInput(
        sender="customer@example.com",
        subject="Invoice INV-123 already paid",
        body="Hello, we already paid invoice INV-123 last week. Please check."
    )

    # 2. Simulated AI outputs (hardcoded for now)
    case_type = "PAYMENT_ISSUE"
    confidence = 0.9
    invoice_ids = ["INV-123"]
    amount = None
    rationale = "Customer claims invoice already paid"

    # 3. Run decision logic
    decision = make_decision(
        case_type=case_type,
        confidence=confidence,
        invoice_ids=invoice_ids,
        amount=amount,
        rationale=rationale,
    )

    # 4. Output result
    print("Email received:")
    print(email.model_dump())
    print("\nFinal decision:")
    print(decision.model_dump())


if __name__ == "__main__":
    main()
