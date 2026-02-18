import os
from cerebras.cloud.sdk import Cerebras


def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set")

    return Cerebras(api_key=api_key)


def call_llm(prompt: str) -> str:
    """
    Calls Cerebras with a prompt and returns raw text output.
    """
    client = get_cerebras_client()

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
