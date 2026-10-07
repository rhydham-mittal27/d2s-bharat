"""JWT authentication: register, login, current user, and the guard that protects the API.

* Passwords: Argon2id via pwdlib (salted, memory-hard). Login runs a hash verification even for
  unknown emails, so response time does not reveal which emails have accounts.
* Tokens: HS256 JWTs signed with ``D2S_JWT_SECRET``; claims sub (user id), email, name, iat, exp.
  Verification is stateless (no database round trip per request, which matters with a remote
  Supabase). Trade-off: a deactivated user's token stays valid until it expires (12 h by default).
* Guard: an app-wide dependency. Every /api route needs ``Authorization: Bearer <token>`` except the
  public ones below. ``D2S_AUTH_REQUIRED=false`` switches it off (tests, offline demos).
"""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pwdlib import PasswordHash
from pydantic import BaseModel, Field, field_validator

from d2s.api.db import DB
from d2s.config import get_settings
from d2s.db import Conflict, NotFound, UserRepository, session_scope
from d2s.db.base import utcnow

log = logging.getLogger(__name__)

ALGORITHM = "HS256"
PUBLIC_PATHS = {"/api/health", "/api/auth/register", "/api/auth/login", "/api/auth/token"}
MIN_SECRET_LEN = 32

hasher = PasswordHash.recommended()
_DUMMY_HASH = hasher.hash("timing-equaliser-not-a-password")
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)  # enables /docs "Authorize"

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---- models ----------------------------------------------------------------------------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("enter a valid email address")
        return v


class RegisterBody(Credentials):
    name: str = Field(..., min_length=1, max_length=120)
    password: str = Field(..., min_length=8, max_length=128, description="at least 8 characters")

    @field_validator("password")
    @classmethod
    def _strength(cls, v: str) -> str:
        if v.strip() != v or len(set(v)) < 4:
            raise ValueError("choose a stronger password (no leading/trailing spaces, varied characters)")
        return v


class UserOut(BaseModel):
    id: str
    email: str
    name: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserOut


# ---- tokens ----------------------------------------------------------------------------------------------
def _secret() -> str:
    s = get_settings().jwt_secret
    if not s or len(s) < MIN_SECRET_LEN:
        raise HTTPException(503, f"server auth is not configured: set D2S_JWT_SECRET (>= {MIN_SECRET_LEN} chars)")
    return s


def issue_token(user: UserOut) -> TokenOut:
    now = datetime.now(UTC)
    exp = now + timedelta(minutes=get_settings().jwt_ttl_minutes)
    claims = {"sub": user.id, "email": user.email, "name": user.name, "iat": now, "exp": exp, "typ": "access"}
    return TokenOut(access_token=jwt.encode(claims, _secret(), algorithm=ALGORITHM), expires_at=exp, user=user)


def decode_token(token: str) -> UserOut:
    try:
        c = jwt.decode(token, _secret(), algorithms=[ALGORITHM], options={"require": ["sub", "exp", "iat"]})
    except jwt.ExpiredSignatureError:
        raise _unauthorized("session expired; sign in again") from None
    except jwt.InvalidTokenError:
        raise _unauthorized("invalid token") from None
    if c.get("typ") != "access":
        raise _unauthorized("invalid token")
    return UserOut(id=c["sub"], email=c.get("email", ""), name=c.get("name", ""))


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})


# ---- guard -----------------------------------------------------------------------------------------------
async def auth_guard(request: Request, token: Annotated[str | None, Depends(oauth2)]) -> UserOut | None:
    """App-wide dependency. Stores the user on request.state for the routes that need it."""
    request.state.user = None
    if not get_settings().auth_required or request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
        return None
    if not token:
        raise _unauthorized("sign in to use the API")
    user = decode_token(token)
    request.state.user = user
    return user


def current_user_id(request: Request) -> str | None:
    """Owner for user-scoped data; None when auth is switched off."""
    u = getattr(request.state, "user", None)
    return u.id if u else None


OwnerId = Annotated[str | None, Depends(current_user_id)]


# ---- routes ----------------------------------------------------------------------------------------------
def _users_db():
    if not DB.enabled:
        raise HTTPException(503, "accounts need the database; set D2S_DATABASE_URL")
    return session_scope(DB.factory)


def _out(r) -> UserOut:
    return UserOut(id=r.id, email=r.email, name=r.name)


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: RegisterBody):
    pw_hash = hasher.hash(body.password)  # hash before opening the transaction (it is deliberately slow)
    try:
        with _users_db() as s:
            user = _out(UserRepository(s).create(body.email, body.name, pw_hash))
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    log.info("registered user %s", user.id)
    return issue_token(user)


def _authenticate(email: str, password: str) -> UserOut:
    with _users_db() as s:
        r = UserRepository(s).by_email(email)
        ok = hasher.verify(password, r.password_hash if r else _DUMMY_HASH)
        if not (r and ok and r.is_active):
            raise _unauthorized("wrong email or password")
        r.last_login_at = utcnow()
        return _out(r)


@router.post("/login", response_model=TokenOut)
def login(body: Credentials):
    return issue_token(_authenticate(body.email, body.password))


@router.post("/token", response_model=TokenOut, include_in_schema=True)
def token_form(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """OAuth2 password form (username = email), used by the Swagger UI "Authorize" button."""
    return issue_token(_authenticate(form.username.strip().lower(), form.password))


@router.get("/me", response_model=UserOut)
def me(request: Request):
    u = getattr(request.state, "user", None)
    if u is None:
        raise _unauthorized("not signed in")
    with _users_db() as s:  # confirm the account still exists and is active
        try:
            r = UserRepository(s).get(u.id)
        except NotFound:
            raise _unauthorized("account no longer exists") from None
        if not r.is_active:
            raise _unauthorized("account disabled")
        return _out(r)
