import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

# API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Gemini client
client = genai.Client(api_key=api_key)

# Image path
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "images3.jpg"

# Read image
with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

# Convert image to base64
image_data = base64.b64encode(image_bytes).decode("utf-8")

# VLM prompt
prompt = """
Analyze this construction-site image for a Construction Intelligence Copilot.

Describe:
1. What is visible in the image
2. Construction activities taking place
3. Materials, equipment, or machinery visible
4. Workers and their activities
5. Structural elements visible
6. Any visible safety conditions or potential hazards
7. Overall construction progress or site condition

Be factual. Do not invent details that cannot be observed.
"""

# Gemini Interactions API
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=[
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image",
            "data": image_data,
            "mime_type": "image/jpeg"
        }
    ]
)

print("\n==============================")
print("GEMINI VLM TEST")
print("==============================")
print(f"Image: {IMAGE_PATH.name}")
print("\nGemini Analysis:\n")
print(interaction.output_text)
print("\n==============================")
