from google import genai
import os
from dotenv import load_dotenv


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

gemini_client = genai.Client(api_key=gemini_api_key)