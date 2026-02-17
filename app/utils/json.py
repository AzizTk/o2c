import json
import re


def extract_json(text: str) -> dict:
    """
    Extracts the first JSON object found in a string.
    Raises ValueError if no valid JSON is found.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON block from text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")

    return json.loads(match.group(0))
