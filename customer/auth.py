"""Verify Firebase ID tokens using Google's signed public certificates.

Ordinary ID-token verification needs a project ID and public certificates, not
an Admin credential. Revocation checks additionally call Firebase's Admin API
and are opt-in through FIREBASE_CHECK_REVOKED.
"""
import math
import re
import secrets
import threading
import time

import httpx
from google.auth import jwt

CERTIFICATES_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"


class InvalidSession(ValueError):
    """The provided ID token is invalid, expired, revoked or disabled."""


class VerificationUnavailable(RuntimeError):
    """The verifier could not contact a required authentication service."""


class PublicCertificates:
    def __init__(self, fetch=None, clock=time.monotonic):
        self._fetch = fetch or self._download
        self._clock = clock
        self._certificates = {}
        self._expires = 0
        self._refreshed = float("-inf")
        self._lock = threading.RLock()

    @staticmethod
    def _download():
        # This fixed Google URL never comes from an untrusted token or request.
        try:
            response = httpx.get(CERTIFICATES_URL, timeout=10, follow_redirects=False)
            response.raise_for_status()
            return response.json(), response.headers.get("cache-control", "")
        except (httpx.HTTPError, ValueError):
            raise VerificationUnavailable("Google sign-in verification is temporarily unavailable.") from None

    def get(self, key_id):
        with self._lock:
            now = self._clock()
            expired = now >= self._expires
            # Refresh unknown keys after rotation, but bound refreshes for bad IDs.
            rotated = key_id not in self._certificates and now - self._refreshed >= 30
            if expired or rotated:
                certificates, cache_control = self._fetch()
                if (not isinstance(certificates, dict) or not certificates or
                        any(not isinstance(k, str) or not isinstance(v, str) or
                            "BEGIN CERTIFICATE" not in v for k, v in certificates.items())):
                    raise VerificationUnavailable("Google sign-in certificates could not be loaded.")
                max_age = re.search(r"(?:^|,)\s*max-age=(\d+)", cache_control, re.I)
                self._certificates = certificates
                self._refreshed = now
                self._expires = now + min(int(max_age.group(1)), 86400) if max_age else now + 300
            return self._certificates

    def cached(self):
        with self._lock:
            return bool(self._certificates) and self._clock() < self._expires


class FirebaseTokenVerifier:
    def __init__(self, project_id, check_revoked=False, certificates=None, revocation_checker=None):
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("A Firebase project ID is required.")
        self.project_id = project_id
        self.check_revoked = check_revoked
        self.certificates = certificates or PublicCertificates()
        self._revocation_checker = revocation_checker
        self._admin_app = None
        self._admin_lock = threading.RLock()

    def diagnostic(self):
        return {"configured": True, "method": "firebase-signed-id-token",
                "revocationChecks": self.check_revoked,
                "publicCertificatesCached": self.certificates.cached()}

    def _check_revocation(self, token):
        if self._revocation_checker is not None:
            return self._revocation_checker(token)
        import firebase_admin
        from firebase_admin import auth
        try:
            with self._admin_lock:
                if self._admin_app is None:
                    self._admin_app = firebase_admin.initialize_app(
                        options={"projectId": self.project_id},
                        name="grounded-revocation-" + secrets.token_hex(6))
            return auth.verify_id_token(token, app=self._admin_app, check_revoked=True)
        except (auth.InvalidIdTokenError, auth.UserDisabledError):
            raise InvalidSession("This sign-in session is no longer valid.") from None
        except Exception:
            # A credential or network failure is a server problem, not bad login.
            raise VerificationUnavailable("Server-side sign-in verification is temporarily unavailable.") from None

    def __call__(self, token):
        if not isinstance(token, str) or not token or len(token) > 16384:
            raise InvalidSession("Invalid Firebase ID token.")
        try:
            header = jwt.decode_header(token)
            if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str) or not header["kid"]:
                raise InvalidSession("Invalid Firebase ID token header.")
        except InvalidSession:
            raise
        except Exception:
            raise InvalidSession("Invalid Firebase ID token.") from None
        certificates = self.certificates.get(header["kid"])
        if header["kid"] not in certificates:
            raise InvalidSession("Unknown Firebase signing key.")
        try:
            payload = jwt.decode(token, certs=certificates, audience=self.project_id)
            now = time.time()
            if payload.get("iss") != "https://securetoken.google.com/" + self.project_id:
                raise InvalidSession("Invalid Firebase token issuer.")
            uid = payload.get("sub")
            if not isinstance(uid, str) or not uid or len(uid) > 128:
                raise InvalidSession("Invalid Firebase token subject.")
            for claim in ("exp", "iat", "auth_time"):
                value = payload.get(claim)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise InvalidSession("Invalid Firebase token timestamp.")
            if payload["exp"] <= now or payload["iat"] > now or payload["auth_time"] > now:
                raise InvalidSession("Expired or premature Firebase token.")
            payload["uid"] = uid
        except InvalidSession:
            raise
        except Exception:
            raise InvalidSession("Invalid Firebase ID token signature or claims.") from None
        if self.check_revoked:
            self._check_revocation(token)
        return payload
