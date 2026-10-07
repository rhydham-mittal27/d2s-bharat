"""JWT auth: register/login, the API guard, token tampering, and per-user data isolation."""

import io
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from d2s.config import get_settings
from d2s.db import create_schema
from tests_support import artifacts_ready

pytestmark = pytest.mark.skipif(not artifacts_ready(), reason="analysis outputs / artifacts not built")

PW = "correct-horse-42"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from d2s.api.db import DB
    from d2s.api.main import app

    settings = get_settings()
    with TestClient(app) as c:
        settings.auth_required = True
        DB.configure("sqlite://")
        create_schema(DB.engine)
        from d2s.db import session_scope
        from d2s.db.seed import seed_courses

        with session_scope(DB.factory) as s:
            seed_courses(s)
        yield c
        settings.auth_required = False
        DB.dispose()


def _register(c, email, name="Test User", password=PW):
    return c.post("/api/auth/register", json={"email": email, "name": name, "password": password})


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_register_login_and_me(client):
    r = _register(client, "Asha@Example.com", "Asha")
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["email"] == "asha@example.com" and body["token_type"] == "bearer"
    me = client.get("/api/auth/me", headers=_auth(body["access_token"])).json()
    assert me == body["user"]

    assert _register(client, "asha@example.com").status_code == 409  # duplicate, case-insensitive
    assert _register(client, "not-an-email").status_code == 422
    assert _register(client, "b@example.com", password="short").status_code == 422
    assert _register(client, "b@example.com", password="aaaaaaaaaa").status_code == 422  # too uniform

    ok = client.post("/api/auth/login", json={"email": "ASHA@example.com", "password": PW})
    assert ok.status_code == 200 and ok.json()["user"]["id"] == body["user"]["id"]
    wrong = client.post("/api/auth/login", json={"email": "asha@example.com", "password": "nope-nope-1"})
    unknown = client.post("/api/auth/login", json={"email": "ghost@example.com", "password": PW})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]  # does not reveal which emails exist

    form = client.post("/api/auth/token", data={"username": "asha@example.com", "password": PW})
    assert form.status_code == 200  # Swagger "Authorize" flow


def test_guard_protects_api_but_not_health(client):
    assert client.get("/api/health").status_code == 200
    r = client.get("/api/overview")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert client.post("/api/plan/solve", json={}).status_code == 401
    token = _register(client, "guard@example.com").json()["access_token"]
    assert client.get("/api/overview", headers=_auth(token)).status_code == 200


def test_tampered_expired_and_unsigned_tokens_are_rejected(client):
    token = _register(client, "tamper@example.com").json()["access_token"]
    secret = get_settings().jwt_secret
    claims = jwt.decode(token, secret, algorithms=["HS256"])

    other_key = jwt.encode(claims, "some-other-secret-" + "y" * 40, algorithm="HS256")
    expired = jwt.encode({**claims, "exp": datetime.now(UTC) - timedelta(minutes=1)}, secret, algorithm="HS256")
    unsigned = jwt.encode(claims, None, algorithm="none")
    wrong_type = jwt.encode({**claims, "typ": "refresh"}, secret, algorithm="HS256")
    head, payload, sig = token.split(".")
    flipped = ".".join([head, payload, sig[:-2] + ("A" if sig[-2] != "A" else "B") + sig[-1]])

    for bad in (other_key, expired, unsigned, wrong_type, flipped, "garbage"):
        assert client.get("/api/overview", headers=_auth(bad)).status_code == 401, bad
    assert "expired" in client.get("/api/overview", headers=_auth(expired)).json()["detail"]


def test_users_only_see_their_own_cohorts_and_runs(client):
    a = _register(client, "owner-a@example.com").json()["access_token"]
    b = _register(client, "owner-b@example.com").json()["access_token"]
    csv = ("id,coding_skills,maths_stats_skills,ai_and_ml_skills,big_data_skills,dashboard_and_storytelling_skills\n"
           "x,2,2,2,2,2\ny,3,3,3,3,3\n")
    prof = client.post("/api/cohort/profile", headers=_auth(a),
                       files={"file": ("mine.csv", io.BytesIO(csv.encode()), "text/csv")}).json()
    cid = prof["cohort_id"]
    run = client.post("/api/plan/full", headers=_auth(a), json={"cohort_id": cid, "label": "A's plan"}).json()
    rid = run["run_id"]

    assert [c["id"] for c in client.get("/api/cohorts", headers=_auth(a)).json()] == [cid]
    assert client.get(f"/api/plan/runs/{rid}", headers=_auth(a)).json()["label"] == "A's plan"

    # B sees none of it, and A's ids behave like missing ids
    assert client.get("/api/cohorts", headers=_auth(b)).json() == []
    assert client.get("/api/plan/runs", headers=_auth(b)).json() == []
    assert client.get(f"/api/cohorts/{cid}", headers=_auth(b)).status_code == 404
    assert client.get(f"/api/plan/runs/{rid}", headers=_auth(b)).status_code == 404
    assert client.delete(f"/api/plan/runs/{rid}", headers=_auth(b)).status_code == 404
    assert client.post("/api/plan/solve", headers=_auth(b), json={"cohort_id": cid}).status_code == 404
    assert client.get(f"/api/plan/runs/{rid}", headers=_auth(a)).status_code == 200  # untouched


def test_missing_secret_is_reported_not_crashing(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "jwt_secret", "short")
    r = client.post("/api/auth/login", json={"email": "owner-a@example.com", "password": PW})
    assert r.status_code == 503 and "D2S_JWT_SECRET" in r.json()["detail"]
