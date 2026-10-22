"""Create one private FDR-scoped HTTP credential, without printing the secret."""

import argparse
import json
import os
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".private/modal-proxy.json")
    args = parser.parse_args()
    path = Path(args.output)
    workspace = modal.Workspace.from_context()
    if path.exists():
        credential = json.loads(path.read_text())
        found = any(
            t.token_id == credential["token_id"]
            for t in workspace.proxy_tokens.list(environment_name="FDR")
        )
        if not found:
            raise ValueError("Saved proxy token does not have FDR access")
        print(
            json.dumps({"status": "existing", "environment": "FDR", "credential_file": str(path)})
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = workspace.proxy_tokens.create()
    try:
        metadata = next(t for t in workspace.proxy_tokens.list() if t.token_id == token.token_id)
        if not metadata.scoped:
            raise RuntimeError(
                "Workspace lacks scoped proxy tokens; refusing workspace-wide credential"
            )
        workspace.proxy_tokens.allow(token.token_id, "FDR")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "token_id": token.token_id,
                    "token_secret": token.token_secret,
                    "environment": "FDR",
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        workspace.proxy_tokens.delete(token.token_id)
        raise
    print(
        json.dumps(
            {
                "status": "created",
                "environment": "FDR",
                "credential_file": str(path),
                "secret_printed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
