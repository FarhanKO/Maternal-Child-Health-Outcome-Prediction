"""
Create a extracted_features.txt in the same directory of this code and run it.
It will get all the features from the csv and write them in the txt file.
"""
from pathlib import Path
import csv

DATASET_PATH = Path(__file__).with_name("maternal_child_dataset_project_ready.csv")
OUTPUT_PATH = Path(__file__).with_name("extracted_features.txt")


def extract_features(csv_path: Path = DATASET_PATH) -> list[str]:
    """Read the CSV header row and return every column as a feature name."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader, [])

    return header


