"""Read-only verification of a downloaded run, including retained partial attempts."""

import argparse
import json
from pathlib import Path

from k2_interp.artifacts import file_hash
from k2_interp.schema import digest


def verify(root: Path):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    expected = digest({"spec": manifest["spec"], "identity": manifest["identity"]})
    if expected != manifest["request_sha256"]:
        raise ValueError("Run manifest request/identity hash mismatch")
    checked = set()

    def files(hashes):
        for relative, expected_hash in hashes.items():
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise ValueError("Receipt path escapes run")
            if not path.is_file() or file_hash(path) != expected_hash:
                raise ValueError(f"Trace checksum mismatch or missing file: {relative}")
            checked.add(relative)

    complete, partial = [], []
    for sample in manifest["spec"]["samples"]:
        from k2_interp.schema import identifier

        base = root / "samples" / identifier(sample["id"])
        receipt_path = base / "complete.json"
        completed_attempt = None
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "complete":
                raise ValueError("Invalid completion receipt")
            completed_attempt = (root / receipt["attempt"]).resolve()
            if not completed_attempt.is_relative_to(base.resolve()):
                raise ValueError("Completed attempt escapes sample")
            result_relative = str((completed_attempt / "result.json").relative_to(root))
            if result_relative not in receipt["files"]:
                raise ValueError("Completed sample has no receipted result")
            files(receipt["files"])
            complete.append(sample["id"])
        for attempt in base.glob("attempt-*"):
            if attempt.resolve() == completed_attempt:
                continue
            progress = attempt / "progress.json"
            if progress.is_file():
                files(json.loads(progress.read_text())["files"])
            partial.append(
                {"path": str(attempt.relative_to(root)), "has_progress_receipt": progress.is_file()}
            )
    return {
        "hashes_verified": True,
        "verified_files": len(checked),
        "completed_samples": complete,
        "expected_samples": len(manifest["spec"]["samples"]),
        "run_complete": len(complete) == len(manifest["spec"]["samples"]),
        "retained_partial_attempts": partial,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = verify(args.run)
    print(json.dumps(result, indent=2))
    if not result["run_complete"] and not args.allow_partial:
        raise SystemExit(
            "Available receipt hashes pass, but some samples have no completion receipt."
        )


if __name__ == "__main__":
    main()
