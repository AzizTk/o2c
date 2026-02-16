from pydantic import BaseModel
from typing import Optional, List, Literal


class Decision(BaseModel):
    case_type: Literal[
        "PAYMENT_ISSUE",
        "DEDUCTION",
        "DISPUTE",
        "UNKNOWN"
    ]

    invoice_ids: Optional[List[str]] = None
    amount: Optional[float] = None
    confidence: float

    action: Literal[
        "AUTO_REPLY",
        "QUEUE_FOR_REVIEW"
    ]

    rationale: str
