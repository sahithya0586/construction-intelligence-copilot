import os
import base64
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


# ============================================================
# 2. IMAGE DIRECTORIES
# ============================================================

IMAGE_DIR = PROJECT_ROOT / "data" / "images"

OUTPUT_DIR = PROJECT_ROOT / "data" / "image_metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. PROMPT
# ============================================================

PROMPT = """
You are a construction-site intelligence assistant.

Analyze this construction-site image carefully.

Return ONLY valid JSON with the following structure:

{
  "image_summary": "",
  "construction_phase": "",
  "activities": [],
  "materials": [],
  "equipment": [],
  "structural_elements": [],
  "workers": [],
  "safety_hazards": [],
  "site_conditions": [],
  "progress_indicators": []
}

Instructions:

- Describe only what can reasonably be observed.
- Do not invent workers, equipment, materials, or hazards.
- Use short, factual descriptions.
- "construction_phase" should identify the likely construction stage.
- "activities" should describe visible construction work.
- "materials" should contain visible construction materials.
- "equipment" should contain visible machinery/tools.
- "structural_elements" should contain visible building elements.
- "workers" should describe workers only if they are actually visible.
- "safety_hazards" should contain visible or strongly indicated safety risks.
- "site_conditions" should describe the current physical condition of the site.
- "progress_indicators" should describe evidence of construction progress.
"""


# ============================================================
# 4. SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# 5. PROCESS ONE IMAGE
# ============================================================

def analyze_image(image_path):

    print(f"\nAnalyzing: {image_path.name}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # Determine MIME type
    extension = image_path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }

    mime_type = mime_types[extension]

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=[
            {
                "type": "text",
                "text": PROMPT
            },
            {
                "type": "image",
                "data": image_data,
                "mime_type": mime_type
            }
        ]
    )

    response_text = interaction.output_text.strip()

    # Remove markdown code fences if Gemini adds them
    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    try:
        analysis = json.loads(response_text)

    except json.JSONDecodeError:
        print("WARNING: Gemini did not return valid JSON.")
        print(response_text)

        analysis = {
            "image_summary": response_text,
            "construction_phase": "",
            "activities": [],
            "materials": [],
            "equipment": [],
            "structural_elements": [],
            "workers": [],
            "safety_hazards": [],
            "site_conditions": [],
            "progress_indicators": []
        }

    return analysis


# ============================================================
# 6. PROCESS ALL IMAGES
# ============================================================

def main():

    image_files = sorted(
        [
            path
            for path in IMAGE_DIR.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    print("=" * 60)
    print("CONSTRUCTION IMAGE INGESTION")
    print("=" * 60)

    print(f"Image directory: {IMAGE_DIR}")
    print(f"Images found: {len(image_files)}")

    if not image_files:
        print("No images found.")
        return

    successful = 0

    for image_path in image_files:

        try:

            analysis = analyze_image(image_path)

            result = {
                "image_id": image_path.stem,
                "image_filename": image_path.name,
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "analysis": analysis
            }

            output_file = OUTPUT_DIR / f"{image_path.stem}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(f"Saved: {output_file.name}")

            successful += 1

        except Exception as e:

            print(f"ERROR processing {image_path.name}: {e}")

    print("\n" + "=" * 60)
    print("IMAGE INGESTION COMPLETE")
    print("=" * 60)

    print(f"Successful: {successful}/{len(image_files)}")
    print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":
    main()
