from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from .db import get_user_by_username
from .security import verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["Authentication & Clearance"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Verifies operator credentials and returns a cryptographic JWT bearer token."""
    user = get_user_by_username(req.username.strip())
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Invalid Operator Callsign or ID",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(req.password, user["password_hash"], user["salt"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Invalid Security Clearance Password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = {
        "sub": user["username"],
        "role": user["role"],
        "callsign": user["callsign"]
    }
    token = create_access_token(token_data)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "role": user["role"],
            "callsign": user["callsign"]
        }
    }

@router.get("/me")
def get_current_profile(user: dict = Depends(get_current_user)):
    """Returns the authenticated operator profile and clearance role."""
    return {
        "username": user.get("sub"),
        "role": user.get("role"),
        "callsign": user.get("callsign"),
        "status": "CLEARANCE_ACTIVE"
    }

@router.get("/roles")
def get_roles():
    """Lists available system roles."""
    return {
        "roles": [
            {"id": "COMMANDER", "title": "Fleet Commander", "scope": "Full Command & Interdiction Priority"},
            {"id": "ANALYST", "title": "Intelligence Analyst", "scope": "LSTM Deep Anomaly & Trajectory Tensors"},
            {"id": "OPERATOR", "title": "Tactical Operator", "scope": "Live Radar Monitoring & Sensor Controls"}
        ]
    }
