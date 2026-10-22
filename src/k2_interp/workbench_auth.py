"""SDK-issued one-use login tickets and short-lived, signed HttpOnly browser sessions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

COOKIE = "k2_workbench"


class BrowserAuth:
    def __init__(self, signing_key, tickets, secure=True):
        if len(signing_key) < 32:
            raise ValueError("Workbench signing key must contain at least 32 characters")
        self.key, self.tickets, self.secure = signing_key.encode(), tickets, secure

    def sign(self, body):
        return hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()

    def issue(self):
        body = f"{int(time.time()) + 43200}.{secrets.token_hex(16)}"
        return f"{body}.{self.sign(body)}"

    def valid(self, cookie):
        try:
            expires, nonce, signature = cookie.split(".")
            return int(expires) > time.time() and hmac.compare_digest(
                signature, self.sign(expires + "." + nonce)
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def redeem(self, ticket):
        if not isinstance(ticket, str) or not 32 <= len(ticket) <= 128:
            return False
        key = "ticket:" + hashlib.sha256(ticket.encode()).hexdigest()
        # Dict.pop is an atomic consume, so concurrent redemption cannot reuse a ticket.
        value = self.tickets.pop(key, None)
        return isinstance(value, dict) and value.get("expires", 0) > time.time()
