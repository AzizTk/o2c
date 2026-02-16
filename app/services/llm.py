import os
import google.generativeai as genai


def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name="gemini-2.5-flash"
    )


def call_llm(prompt: str) -> str:
    """
    Calls Gemini with a prompt and returns raw text output.
    """
    model = get_gemini_model()
    response = model.generate_content(prompt)
    return response.text
