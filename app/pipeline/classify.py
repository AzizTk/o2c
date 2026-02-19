import json
import re
from models.input import EmailInput
from services.llm import call_llm
from dataclasses import dataclass
from typing import Literal

ALLOWED_CASES = {"Payment Claim", "Dispute", "General AR Request"}

@dataclass
class ClassificationResult:
    case_type: Literal["Payment Claim", "Dispute", "General AR Request"]
    confidence: float
    rationale: str
    raw_llm_output: str


def extract_json(raw: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    Handles markdown fences and extra text.
        args:
            raw: The raw string output from the LLM, which may contain markdown fences and extra text.

        returns:
            A dictionary parsed from the first JSON object found in the input string.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group())


def classify_email(email: EmailInput) -> ClassificationResult:

    """Classifies an email into a case type with confidence.

        args:
            email: EmailInput object containing subject and body
            
        returns:
            ClassificationResult with case_type, confidence, rationale, and raw LLM output
        
    """
    prompt = f"""
You are classifying customer finance emails.
Classify the email into exactly one of:
    1. Payment Claim (customer says they paid)
    2. Dispute (customer claims an issue / short pay / wrong invoice)
    3. General AR Request (copy of invoice, statement request, change billing details)
Each case should include a recommended next step.
Return ONLY valid JSON in this format:
{{
  "case_type": "ONE_OF_THE_ABOVE",
  "confidence": number between 0 and 1,
  "rationale": "short explanation"
  "Recommended next step: (e.g. 'Route to Cash Application queue', 'Route to Disputes team', 'Route to AR Support for manual review')"
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


