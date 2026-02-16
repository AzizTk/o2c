import json
import re
from models.input import EmailInput
from services.llm import call_llm


ALLOWED_CASES = {"PAYMENT_ISSUE", "DEDUCTION", "DISPUTE", "UNKNOWN"}



def extract_json(raw: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    Handles markdown fences and extra text.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group())

def classify_email(email: EmailInput) -> tuple[str, float, str]:
    """Classifies the email and returns a tuple of (case_type, confidence, rationale).
    args:
    - email: EmailInput object containing subject and body of the email.
    returns:
    - case_type: One of PAYMENT_ISSUE, DEDUCTION, DISPUTE, UNKNOWN
    - confidence: A float between 0 and 1 indicating the model's confidence in the classification.
    - rationale: A short explanation of why the model classified the email as it did.
    """

    prompt = f"""
You are classifying customer finance emails.
Classify the email into exactly one of:
- PAYMENT_ISSUE
- DEDUCTION
- DISPUTE
- UNKNOWN
Return ONLY valid JSON in this format:
{{
  "case_type": "ONE_OF_THE_ABOVE",
  "confidence": number between 0 and 1,
  "rationale": "short explanation"
}}
Email:
Subject: {email.subject}
Body: {email.body}
"""

    try:
        raw = call_llm(prompt)
        print("\n--- RAW LLM RESPONSE ---")
        print(raw)
        print("--- END RAW RESPONSE ---\n")

        data = extract_json(raw)

        case_type = data.get("case_type", "UNKNOWN")
        confidence = float(data.get("confidence", 0.0))
        rationale = data.get("rationale", "No rationale provided")

        if case_type not in ALLOWED_CASES:
            case_type = "UNKNOWN"
            confidence = 0.5
            rationale = "Invalid case_type from model"

        return case_type, confidence, rationale

    except Exception as e:
        print("\n--- LLM ERROR ---")
        print(type(e), e)
        print("--- END LLM ERROR ---\n")
        raise

