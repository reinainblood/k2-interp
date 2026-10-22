"""Open a private browser session using current Modal SDK auth. Never print the ticket."""

import argparse
import hashlib
import secrets
import time
import webbrowser

import modal


def login_ticket():
    ticket = secrets.token_urlsafe(32)
    channel = modal.Dict.from_name("k2-interp-live", environment_name="FDR", create_if_missing=True)
    channel.put(
        "ticket:" + hashlib.sha256(ticket.encode()).hexdigest(), {"expires": time.time() + 90}
    )
    return ticket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    args = parser.parse_args()
    url = (
        args.url
        or modal.Function.from_name(
            "k2-interp-workbench", "web", environment_name="FDR"
        ).get_web_url()
    )
    webbrowser.open(url.rstrip("/") + "/#ticket=" + login_ticket())
    print("Opened private workbench session. Login ticket expires in 90 seconds.")


if __name__ == "__main__":
    main()
