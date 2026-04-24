from fastapi import APIRouter, Depends, HTTPException
from backend.database import get_db
from backend.dependencies import get_current_user
from backend.schemas.auth import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshRequest,
    ForgotPasswordRequest, ResetPasswordRequest, ChangePasswordRequest,
)
from backend.services import auth_service, email_service
from backend.security import hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register")
def register(req: RegisterRequest, conn=Depends(get_db)):
    result, error = auth_service.register_user(
        conn, req.email, req.password, req.first_name, req.last_name, req.phone
    )
    if error:
        raise HTTPException(status_code=400, detail=error)
    try:
        email_service.send_verification_email(req.email, result["verification_token"])
    except Exception:
        pass
    return {"message": "Registration successful. Please verify your email."}


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, conn=Depends(get_db)):
    result, error = auth_service.login_user(conn, req.email, req.password)
    if error:
        raise HTTPException(status_code=401, detail=error)
    return result


@router.post("/logout")
def logout(user=Depends(get_current_user), conn=Depends(get_db)):
    auth_service.logout_user(conn, user["id"])
    return {"message": "Logged out"}


@router.post("/refresh")
def refresh(req: RefreshRequest, conn=Depends(get_db)):
    result, error = auth_service.refresh_access_token(conn, req.refresh_token)
    if error:
        raise HTTPException(status_code=401, detail=error)
    return result


@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, conn=Depends(get_db)):
    token = auth_service.create_password_reset_token(conn, req.email)
    if token:
        try:
            email_service.send_password_reset_email(req.email, token)
        except Exception:
            pass
    return {"message": "If the email exists, a reset link has been sent."}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, conn=Depends(get_db)):
    ok, error = auth_service.reset_password(conn, req.token, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=error)
    return {"message": "Password reset successful"}


@router.get("/verify-email/{token}")
def verify_email(token: str, conn=Depends(get_db)):
    ok = auth_service.verify_email(conn, token)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    return {"message": "Email verified successfully"}


@router.post("/resend-verification")
def resend_verification(user=Depends(get_current_user), conn=Depends(get_db)):
    if user["is_verified"]:
        return {"message": "Already verified"}
    from backend.security import generate_token
    from datetime import datetime, timedelta, timezone
    token = generate_token()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM email_verification_tokens WHERE user_id = %s", (user["id"],))
        cur.execute(
            "INSERT INTO email_verification_tokens (user_id, token, expires_at) VALUES (%s, %s, %s)",
            (user["id"], token, datetime.now(timezone.utc) + timedelta(hours=24)),
        )
        conn.commit()
    try:
        email_service.send_verification_email(user["email"], token)
    except Exception:
        pass
    return {"message": "Verification email sent"}
