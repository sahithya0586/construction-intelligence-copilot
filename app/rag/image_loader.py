from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_images():
    images = []

    for image_path in sorted(IMAGES_DIR.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append({
                "filename": image_path.name,
                "path": str(image_path),
                "type": image_path.suffix.lower()
            })

    return images


if __name__ == "__main__":
    images = load_images()

    print("\n==============================")
    print("CONSTRUCTION IMAGE INGESTION")
    print("==============================")
    print(f"Images found: {len(images)}")

    for image in images:
        print(f"- {image['filename']} ({image['type']})")
