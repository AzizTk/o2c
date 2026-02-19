from dataclasses import dataclass
from typing import List, Optional
from models.input import EmailInput
from pipeline.classify import ClassificationResult
from services.llm import call_llm
from utils.json import extract_json



@dataclass
class ExtractionResult:
    customer: Optional[str]
    invoice_ids: List[str]
    amount: Optional[float]
    currency: Optional[str]
    dispute_reason: Optional[str]
    raw_llm_output: str


def extract_fields(
    email: EmailInput,
    classification: ClassificationResult,
    MAX_LLM_RETRIES : int
) -> ExtractionResult:
    """Extracts structured fields from an email based on its classification.

        args:
            email: EmailInput object containing sender, subject, and body
            classification: ClassificationResult object containing case_type and confidence
            MAX_LLM_RETRIES: int, number of times to retry LLM call if JSON extraction fails

        returns:
            ExtractionResult with extracted fields and raw LLM output
    """
    prompt = f"""
You are extracting structured finance information from a customer email.

The email has already been classified as: {classification.case_type}

Extract the following fields when present:
- invoice_ids (array of strings)
- amount (number or null)
- currency (string or null)
- dispute_reason (string or null)
- customer 

Return ONLY valid JSON in this format:
{{
  "customer": "customer name or identifier/email",
  "invoice_ids": [],
  "amount": null,
  "currency": null,
  "dispute_reason": null
}}

Email:   {email.sender}
Subject: {email.subject}
Body: {email.body}
"""
    for attempt in range(MAX_LLM_RETRIES):
        raw = call_llm(prompt)
        try:
            data = extract_json(raw)

            return ExtractionResult(
                invoice_ids=data.get("invoice_ids", []),
                amount=data.get("amount"),
                currency=data.get("currency"),
                dispute_reason=data.get("dispute_reason"),
                customer=data.get("customer"),
                raw_llm_output=raw,
            )
        except Exception:
            if attempt == MAX_LLM_RETRIES - 1:
                raise

