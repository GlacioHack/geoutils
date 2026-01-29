"""Helper function to search for unresolved TODOs in precommit."""

import sys
from pathlib import Path

# Define pattern
PATTERN = "TODO"

# Loop over files
found = False
for path in map(Path, sys.argv[1:]):
    try:
        lines = path.read_text().splitlines()
    except Exception:
        continue

    # Raise error on a found pattern
    for lineno, line in enumerate(lines, start=1):
        if PATTERN in line:
            print(f"{path}:{lineno}: {line.strip()}")
            found = True

# Abort pre-commit if any was found
if found:
    print("\nCommit aborted: TODOs should be resolved or moved to an issue if not finalized in a PR.")
    sys.exit(1)
