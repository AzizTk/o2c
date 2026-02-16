from pydantic import BaseModel


class EmailInput(BaseModel):
    sender: str
    subject: str
    body: str
