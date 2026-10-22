"""Export a downloaded private workbench session without launching a model or contacting Modal."""

import argparse
import json
import os
from pathlib import Path

from k2_interp.conversation_export import conversation_bundle
from k2_interp.telemetry import utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Downloaded artifact-volume root containing checkpoint directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    temporary, metadata = conversation_bundle(
        args.artifacts, session["checkpoint"], session, utc_now()
    )
    try:
        # Exclusive creation prevents overwriting an earlier backup, including through a symlink.
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as destination, temporary.open("rb") as source:
            import shutil

            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "turn_count": metadata["turn_count"],
                "archive_complete": metadata["archive_complete"],
            }
        )
    )
    if not metadata["archive_complete"]:
        raise SystemExit(
            "Export preserved available data; inspect export.json for missing or partial turns."
        )


if __name__ == "__main__":
    main()
