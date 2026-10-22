"""Capture an app-scoped FDR billing receipt without changing any resources."""

import argparse
import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path

from k2_interp.artifacts import atomic_json
from k2_interp.telemetry import utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", nargs="+", required=True)
    parser.add_argument("--output", default=".private/workbench-billing.json")
    args = parser.parse_args()
    result = subprocess.run(
        [
            "modal",
            "billing",
            "report",
            "--for",
            "today",
            "--resolution",
            "h",
            "--tz",
            "local",
            "--show-resources",
            "--json",
        ],
        env={**os.environ, "MODAL_ENVIRONMENT": "FDR"},
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        row
        for row in json.loads(result.stdout)
        if row.get("environment") == "FDR" and row.get("object_id") in args.apps
    ]
    receipt = {
        "observed_at": utc_now(),
        "billing_is_partial": True,
        "period": "today (local time)",
        "cost_usd_reported": str(sum((Decimal(row["cost"]) for row in rows), Decimal(0))),
        "rows": rows,
    }
    atomic_json(Path(args.output), receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
