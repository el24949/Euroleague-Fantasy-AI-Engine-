"""Slice 7: write the validated, role-assigned squad to data/my_team.csv.

Pure file I/O -- no Streamlit, no optimizer. app.py builds the 11 rows and
calls save_my_team(); nothing here decides anything about the lineup.

FORMAT (matches the existing data/my_team.csv, read by
optimizer.my_team.load_roster_csv with comment='#'):
  - the leading '#' comment block is copied from the existing file verbatim
  - header `player_name,role,captain`
  - players: "SURNAME, Name" (quoted -- csv QUOTE_MINIMAL quotes it because of
    the comma), role starter|6th_man|bench, captain=Y on exactly one starter
  - coach: `First Last` (no comma, so unquoted), role coach
  - exactly 11 rows: 5 starter, 1 6th_man, 4 bench, 1 coach; LF line endings

SAFETY: the first save copies the existing file to my_team.csv.bak (never
overwriting an existing .bak); the write goes to a temp file in the same
directory, is fsynced, then os.replace()d onto the real file, so a crash
mid-write can never leave a truncated my_team.csv.
"""

import csv
import io
import os
import shutil
import tempfile
from pathlib import Path

MY_TEAM_PATH = Path(__file__).resolve().parent / "data" / "my_team.csv"
HEADER = ["player_name", "role", "captain"]
ROLE_COUNTS = {"starter": 5, "6th_man": 1, "bench": 4, "coach": 1}
ROLE_ORDER = {"starter": 0, "6th_man": 1, "bench": 2, "coach": 3}


def validate_rows(rows):
    """rows: list of dicts with player_name/role/captain ('Y' or ''). Fails
    loudly rather than writing a file the pipeline would reject."""
    counts = {}
    for row in rows:
        counts[row["role"]] = counts.get(row["role"], 0) + 1
    if counts != ROLE_COUNTS:
        raise ValueError(f"Role counts {counts} != required {ROLE_COUNTS}")
    if any(not row["player_name"].strip() for row in rows):
        raise ValueError("A row has an empty player_name")
    captains = [row for row in rows if row["captain"] == "Y"]
    if len(captains) != 1 or captains[0]["role"] != "starter":
        raise ValueError("Need exactly one captain, and they must be a starter")
    if any(row["captain"] not in ("Y", "") for row in rows):
        raise ValueError("captain must be 'Y' or blank")


def render_csv(rows, header_lines):
    """The exact text that will be written (also used for the UI preview)."""
    ordered = sorted(rows, key=lambda r: ROLE_ORDER[r["role"]])
    buf = io.StringIO()
    buf.write("".join(line + "\n" for line in header_lines))
    writer = csv.writer(buf, lineterminator="\n")  # QUOTE_MINIMAL
    writer.writerow(HEADER)
    for row in ordered:
        writer.writerow([row["player_name"], row["role"], row["captain"]])
    return buf.getvalue()


def read_header_comments(path=MY_TEAM_PATH):
    """The leading run of '#' lines of the existing file (empty list if none)."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            lines.append(line.rstrip("\r\n"))
    return lines


def save_my_team(rows, path=MY_TEAM_PATH):
    """Validate, back up once, then atomically replace `path`. Returns the path."""
    path = Path(path)
    validate_rows(rows)
    text = render_csv(rows, read_header_comments(path))

    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return path
