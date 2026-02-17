import json
from pathlib import Path
from typing import List
from models.input import EmailInput

def load_emails_from_json(path: str) -> List[EmailInput]:
    raw = json.loads(Path(path).read_text())

    if "emails" not in raw or not isinstance(raw["emails"], list):
        raise ValueError("Expected JSON with key 'emails' containing a list")

    emails = []

    for item in raw["emails"]:
        emails.append(
            EmailInput(
                sender=item["from"],
                subject=item["subject"],
                body=item["body"],
                metadata={
                    "id": item.get("id"),
                    "receivedAt": item.get("receivedAt"),
                },
            )
        )

    return emails
