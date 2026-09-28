from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "data" / "images"


def get_image_paths():
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    image_paths = []

    for image_path in sorted(IMAGES_DIR.iterdir()):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in supported_extensions
        ):
            image_paths.append(image_path)

    return image_paths


def process_image(image_path):
    """
    Placeholder for VLM processing.

    Later, the VLM will analyze the image
    and return a textual description.
    """

    return {
        "image": image_path.name,
        "description": None,
        "status": "VLM processing pending"
    }


if __name__ == "__main__":
    images = get_image_paths()

    print("\n==============================")
    print("VLM IMAGE PROCESSOR")
    print("==============================")

    print(f"Images found: {len(images)}")

    for image_path in images:
        result = process_image(image_path)

        print(f"\nImage: {result['image']}")
        print(f"Status: {result['status']}")
