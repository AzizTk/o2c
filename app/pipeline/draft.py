from services.llm import call_llm
from models.input import EmailInput
from pipeline.classify import ClassificationResult
from pipeline.extract import ExtractionResult


def generate_draft_response(
    email: EmailInput,
    classification: ClassificationResult,
    extracted: ExtractionResult,
) -> str:
    """
    Generates a draft email response for human review.
    """

    prompt = f"""
You are an Accounts Receivable assistant at Transformance AI. Your name is Aziz Taktak. your e-mail is aziztaktak@transformance.ai

Draft a professional, polite response email based on the information below.
Do NOT invent facts. If information is missing, keep the response generic.
The response should be ready for human review and editing.

Email received:
From: {email.sender}
Subject: {email.subject}
Body:
{email.body}

Classification: {classification.case_type}
Confidence: {classification.confidence}

Extracted information:
- Customer: {extracted.customer}
- Invoice IDs: {extracted.invoice_ids}
- Amount: {extracted.amount}
- Dispute reason: {extracted.dispute_reason}

Return ONLY the email body text. No explanations.
"""

    return call_llm(prompt).strip()
