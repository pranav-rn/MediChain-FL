# scripts/prepare_hospital_data.py
import os
import shutil
from pathlib import Path
import random

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def prepare_hospital_data(base_data_path=None, output_dir=None, splits=[150, 175]):
    """
    Splits the base dataset into hospital-specific directories for 2 hospitals.
    Ensures 'NORMAL' and 'PNEUMONIA' subfolders are created for ImageFolder.
    """
    if base_data_path is None:
        base_data_path = PROJECT_ROOT / "data" / "chest_xray" / "train"
    else:
        base_data_path = Path(base_data_path)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data"
    else:
        output_dir = Path(output_dir)

    normal_images = list((base_data_path / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((base_data_path / "PNEUMONIA").glob("*.jpeg"))

    random.shuffle(normal_images)
    random.shuffle(pneumonia_images)

    print(f"Total NORMAL images available: {len(normal_images)}")
    print(f"Total PNEUMONIA images available: {len(pneumonia_images)}")

    current_normal_idx = 0
    current_pneumonia_idx = 0

    for i, num_samples in enumerate(splits):
        hospital_id = f"hospital_{i + 1}"
        hospital_path = output_dir / hospital_id
        (hospital_path / "NORMAL").mkdir(parents=True, exist_ok=True)
        (hospital_path / "PNEUMONIA").mkdir(parents=True, exist_ok=True)

        # Distribute samples, trying to keep class balance if possible (approx half-half)
        num_normal_for_hospital = num_samples // 2
        num_pneumonia_for_hospital = num_samples - num_normal_for_hospital

        print(
            f"\nPreparing {num_samples} samples for {hospital_id} ({num_normal_for_hospital} NORMAL, {num_pneumonia_for_hospital} PNEUMONIA)..."
        )

        # Copy normal images
        copied_normal = 0
        while copied_normal < num_normal_for_hospital and current_normal_idx < len(
            normal_images
        ):
            shutil.copy(normal_images[current_normal_idx], hospital_path / "NORMAL")
            current_normal_idx += 1
            copied_normal += 1

        # Copy pneumonia images
        copied_pneumonia = 0
        while (
            copied_pneumonia < num_pneumonia_for_hospital
            and current_pneumonia_idx < len(pneumonia_images)
        ):
            shutil.copy(
                pneumonia_images[current_pneumonia_idx], hospital_path / "PNEUMONIA"
            )
            current_pneumonia_idx += 1
            copied_pneumonia += 1

        print(
            f"Finished {hospital_id}: Copied {copied_normal} NORMAL, {copied_pneumonia} PNEUMONIA."
        )

    if current_normal_idx < len(normal_images) or current_pneumonia_idx < len(
        pneumonia_images
    ):
        print(
            f"\nNote: Leftover images in the original dataset: {len(normal_images) - current_normal_idx} NORMAL, {len(pneumonia_images) - current_pneumonia_idx} PNEUMONIA."
        )


if __name__ == "__main__":
    # Clear existing hospital data directories first
    print("Clearing existing hospital data directories...")
    for i in range(1, 3):  # Only for hospital_1 and hospital_2
        hospital_dir = PROJECT_ROOT / "data" / f"hospital_{i}"
        if hospital_dir.exists():
            shutil.rmtree(hospital_dir)

    # Run data preparation. Adjust `base_data_path` if your unzipped structure is different.
    prepare_hospital_data(
        base_data_path=PROJECT_ROOT / "data" / "chest_xray" / "train",
        output_dir=PROJECT_ROOT / "data",
        splits=[150, 175],
    )
    print("\nData preparation complete. You can now run docker-compose up.")
