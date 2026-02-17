from dataclasses import dataclass
from typing import List, Optional
from models.input import EmailInput
from pipeline.classify import ClassificationResult
from services.llm import call_llm
from utils.json import extract_json



@dataclass
class ExtractionResult:
    invoice_ids: List[str]
    amount: Optional[float]
    currency: Optional[str]
    dispute_reason: Optional[str]
    raw_llm_output: str


def extract_fields(
    email: EmailInput,
    classification: ClassificationResult,
) -> ExtractionResult:
    prompt = f"""
You are extracting structured finance information from a customer email.

The email has already been classified as: {classification.case_type}

Extract the following fields when present:
- invoice_ids (array of strings)
- amount (number or null)
- currency (string or null)
- dispute_reason (string or null)

Return ONLY valid JSON in this format:
{{
  "invoice_ids": [],
  "amount": null,
  "currency": null,
  "dispute_reason": null
}}

Email:
Subject: {email.subject}
Body: {email.body}
"""

    raw = call_llm(prompt)
    data = extract_json(raw)

    return ExtractionResult(
        invoice_ids=data.get("invoice_ids", []),
        amount=data.get("amount"),
        currency=data.get("currency"),
        dispute_reason=data.get("dispute_reason"),
        raw_llm_output=raw,
    )
