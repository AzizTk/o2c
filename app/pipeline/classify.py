import json
import re
from models.input import EmailInput
from services.llm import call_llm
from dataclasses import dataclass
from typing import Literal

ALLOWED_CASES = {"PAYMENT_ISSUE", "DEDUCTION", "DISPUTE", "UNKNOWN"}

@dataclass
class ClassificationResult:
    case_type: Literal["PAYMENT_ISSUE", "DEDUCTION", "DISPUTE", "UNKNOWN"]
    confidence: float
    rationale: str
    raw_llm_output: str


def extract_json(raw: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    Handles markdown fences and extra text.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group())


def classify_email(email: EmailInput) -> ClassificationResult:
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

    raw = call_llm(prompt)
    data = extract_json(raw)

    case_type = data.get("case_type", "UNKNOWN")
    confidence = float(data.get("confidence", 0.0))
    rationale = data.get("rationale", "No rationale provided")

    if case_type not in ALLOWED_CASES:
        case_type = "UNKNOWN"
        confidence = 0.5
        rationale = "Invalid case_type from model"

    return ClassificationResult(
        case_type=case_type,
        confidence=confidence,
        rationale=rationale,
        raw_llm_output=raw,
    )


