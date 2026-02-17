from typing import Dict, Optional
from pydantic import BaseModel


class EmailInput(BaseModel):
    sender: str
    subject: str
    body: str
    metadata: Optional[Dict[str, str]] = None
