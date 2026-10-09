"""Real RSA signatures exercise authentication without account credentials."""
import base64
import datetime
import json
import time

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient
from google.auth import crypt, jwt

from customer.app import create_app
from customer.auth import FirebaseTokenVerifier, InvalidSession, PublicCertificates, VerificationUnavailable
from customer.config import Settings


@pytest.fixture(scope="module")
def signing_material():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Authentication unit test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))
    private_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    return crypt.RSASigner.from_string(private_pem, key_id="test-key"), cert_pem


def signed_token(material, **changes):
    now = int(time.time())
    payload = {"aud": "grounded-unit-test", "iss": "https://securetoken.google.com/grounded-unit-test",
               "sub": "signed-alice", "iat": now - 5, "exp": now + 3600, "auth_time": now - 20,
               "email_verified": True}
    payload.update(changes)
    return jwt.encode(material[0], payload).decode()


def verifier(material, **kwargs):
    certificates = PublicCertificates(fetch=lambda: ({"test-key": material[1]}, "public, max-age=3600"))
    return FirebaseTokenVerifier("grounded-unit-test", certificates=certificates, **kwargs)


def test_valid_signed_firebase_token_needs_no_admin_credential(signing_material, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    check = verifier(signing_material)
    decoded = check(signed_token(signing_material))
    assert decoded["uid"] == "signed-alice"
    assert decoded["email_verified"] is True
    assert check.diagnostic() == {"configured": True, "method": "firebase-signed-id-token",
                                  "revocationChecks": False, "publicCertificatesCached": True}


@pytest.mark.parametrize("change", [
    {"aud": "another-project"}, {"iss": "https://securetoken.google.com/another-project"},
    {"sub": ""}, {"sub": "x" * 129}, {"sub": 5}, {"auth_time": None}, {"auth_time": True},
    {"exp": int(time.time()) - 120}, {"iat": int(time.time()) + 120},
    {"auth_time": int(time.time()) + 120}, {"exp": "tomorrow"},
])
def test_required_firebase_claims_are_enforced(signing_material, change):
    with pytest.raises(InvalidSession):
        verifier(signing_material)(signed_token(signing_material, **change))


def test_tampered_signature_and_unsigned_token_are_rejected(signing_material):
    valid = signed_token(signing_material)
    header, payload, signature = valid.split(".")
    data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    data["sub"] = "another-user"
    tampered = base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
    with pytest.raises(InvalidSession):
        verifier(signing_material)(header + "." + tampered + "." + signature)
    no_signature = base64.urlsafe_b64encode(b'{"alg":"none","kid":"test-key"}').decode().rstrip("=")
    with pytest.raises(InvalidSession):
        verifier(signing_material)(no_signature + "." + payload + ".")


def test_certificates_follow_cache_lifetime_and_refresh_rotated_keys(signing_material):
    now, calls = [100], []
    def fetch():
        calls.append(now[0])
        return {"test-key": signing_material[1], "next-key": signing_material[1]}, "max-age=60, public"
    certs = PublicCertificates(fetch=fetch, clock=lambda: now[0])
    certs.get("test-key")
    now[0] = 120
    certs.get("test-key")
    assert calls == [100]
    # Unknown IDs cannot make every request refresh Google's certificates.
    certs.get("unknown")
    assert calls == [100]
    now[0] = 131
    certs.get("unknown")
    assert calls == [100, 131]
    now[0] = 192
    assert not certs.cached()
    certs.get("test-key")
    assert calls == [100, 131, 192]


def test_optional_revocation_check_remains_fail_closed(signing_material):
    checked = []
    def reject(token):
        checked.append(token)
        raise InvalidSession("Revoked.")
    checker = verifier(signing_material, check_revoked=True, revocation_checker=reject)
    token = signed_token(signing_material)
    with pytest.raises(InvalidSession):
        checker(token)
    assert checked == [token] and checker.diagnostic()["revocationChecks"] is True


def test_authentication_outage_is_503_and_does_not_expose_details(tmp_path):
    def unavailable(token):
        raise VerificationUnavailable("Credential path and raw internal details")
    settings = Settings(database=tmp_path / "workspace.sqlite3", firebase={"projectId": "grounded-unit-test"})
    client = TestClient(create_app(settings, token_verifier=unavailable), base_url=settings.origin,
                        headers={"Authorization": "Bearer private-token"})
    response = client.get("/api/session")
    assert response.status_code == 503
    assert response.headers["x-auth-error"] == "verification-unavailable"
    assert "Credential path" not in response.text and "private-token" not in response.text


def test_authenticated_session_and_workspace_use_signed_uid(tmp_path, signing_material):
    settings = Settings(database=tmp_path / "workspace.sqlite3", firebase={"projectId": "grounded-unit-test"})
    client = TestClient(create_app(settings, token_verifier=verifier(signing_material)), base_url=settings.origin,
                        headers={"Authorization": "Bearer " + signed_token(signing_material)})
    response = client.get("/api/session")
    assert response.status_code == 200
    assert response.json()["authenticated"] is True and response.json()["emailVerified"] is True
    assert client.get("/api/workspace").status_code == 200
    client.headers["Authorization"] = "Bearer " + signed_token(signing_material, email_verified=False)
    assert client.get("/api/session").status_code == 403
    client.headers["Authorization"] = "Bearer " + signed_token(signing_material, email_verified="false")
    assert client.get("/api/session").status_code == 403
