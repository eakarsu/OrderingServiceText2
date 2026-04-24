from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from backend.database import get_db
from backend.security import decode_token

security_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    conn=Depends(get_db),
):
    try:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, first_name, last_name, role, phone, is_active, is_verified, created_at FROM users WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    if not row[6]:  # is_active
        raise HTTPException(status_code=403, detail="Account deactivated")

    return {
        "id": row[0],
        "email": row[1],
        "first_name": row[2],
        "last_name": row[3],
        "role": row[4],
        "phone": row[5],
        "is_active": row[6],
        "is_verified": row[7],
        "created_at": str(row[8]),
    }


def require_role(*roles):
    def checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return checker
