"""Verify a downloaded chat/capture ZIP without extracting or executing its contents."""

import argparse
import hashlib
import json
import zipfile
from pathlib import PurePosixPath


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive paths")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names
        ):
            raise ValueError("Unsafe archive path")
        index_name = "capture.json" if "capture.json" in names else "export.json"
        index = json.loads(archive.read(index_name))
        expected = index["files"]
        if set(names) != set(expected) | {index_name}:
            raise ValueError("Archive inventory disagrees with manifest")
        for name, digest in expected.items():
            actual = hashlib.sha256()
            with archive.open(name) as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    actual.update(chunk)
            if actual.hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
        conversation = json.loads(archive.read("conversation.json"))
        return {
            "hashes_verified": True,
            "files": len(expected),
            "turn_count": len(conversation["turns"]),
            "archive_complete": conversation["export"]["archive_complete"],
            "tensor_scope": index.get("scope", "no tensor shards"),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = verify(args.archive)
    print(json.dumps(result, indent=2))
    if not result["archive_complete"] and not args.allow_partial:
        raise SystemExit(
            "Checksums pass, but this chat contains partial or missing results. Inspect coverage."
        )


if __name__ == "__main__":
    main()
