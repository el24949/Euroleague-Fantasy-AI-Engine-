"""Find the most recently uploaded price file (highest round number)."""

import re
from pathlib import Path

FILENAME_PATTERN = re.compile(r"^players_stats26_(\d+)\.xlsx$")


def find_latest_price_file(uploads_dir="data/prices/uploads"):
    """Find the players_stats26_<round>.xlsx file with the highest round number.

    Returns:
        (path, round_number) for the matching file with the highest round number.

    Raises:
        ValueError: if uploads_dir has no files matching players_stats26_<N>.xlsx.
    """
    uploads_path = Path(uploads_dir)

    best_path = None
    best_round = None

    if uploads_path.is_dir():
        for file_path in uploads_path.iterdir():
            match = FILENAME_PATTERN.match(file_path.name)
            if not match:
                continue
            round_number = int(match.group(1))
            if best_round is None or round_number > best_round:
                best_round = round_number
                best_path = file_path

    if best_path is None:
        raise ValueError(
            f"No price files found in '{uploads_dir}'. "
            "Expected filenames like players_stats26_1.xlsx, players_stats26_12.xlsx "
            "(pattern: players_stats26_<round>.xlsx)."
        )

    return best_path, best_round


if __name__ == "__main__":
    path, round_number = find_latest_price_file()
    print(f"Latest price file: {path}")
    print(f"Round number: {round_number}")
