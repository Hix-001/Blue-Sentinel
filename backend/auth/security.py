import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("SENTINEL_JWT_SECRET", "sentinel-c4isr-indo-pacific-secret-key-32b!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security_bearer = HTTPBearer(auto_error=False)

def hash_password(password: str) -> tuple[str, str]:
    """Generates a salt and returns (pbkdf2_hash_hex, salt_hex)."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return key.hex(), salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a password against the stored PBKDF2 hash using constant-time comparison."""
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return secrets.compare_digest(key.hex(), stored_hash)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encodes JWT access token with operator callsign and role."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict:
    """Decodes and validates a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired security clearance token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)) -> dict:
    """Extracts and verifies the user from Authorization: Bearer <token>."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)
