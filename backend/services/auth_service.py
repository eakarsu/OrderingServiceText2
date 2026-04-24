from datetime import datetime, timedelta, timezone
from backend.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token, generate_token


def register_user(conn, email, password, first_name, last_name, phone=""):
    pw_hash = hash_password(password)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM users WHERE email = %s", (email,)
        )
        if cur.fetchone():
            return None, "Email already registered"
        cur.execute(
            """INSERT INTO users (email, password_hash, first_name, last_name, phone)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (email, pw_hash, first_name, last_name, phone),
        )
        user_id = cur.fetchone()[0]
        # Create verification token
        token = generate_token()
        cur.execute(
            """INSERT INTO email_verification_tokens (user_id, token, expires_at)
               VALUES (%s, %s, %s)""",
            (user_id, token, datetime.now(timezone.utc) + timedelta(hours=24)),
        )
        conn.commit()
    return {"user_id": user_id, "verification_token": token}, None


def login_user(conn, email, password):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash, first_name, last_name, role, phone, is_active, is_verified, created_at FROM users WHERE email = %s",
            (email,),
        )
        row = cur.fetchone()
    if not row:
        return None, "Invalid credentials"
    if not row[7]:  # is_active
        return None, "Account deactivated"
    if not verify_password(password, row[2]):
        return None, "Invalid credentials"

    user = {
        "id": row[0], "email": row[1], "first_name": row[3],
        "last_name": row[4], "role": row[5], "phone": row[6],
        "is_active": row[7], "is_verified": row[8], "created_at": str(row[9]),
    }
    access_token = create_access_token({"sub": user["id"], "role": user["role"]})
    refresh_token = create_refresh_token({"sub": user["id"]})

    # Store refresh token
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO refresh_tokens (user_id, token, expires_at)
               VALUES (%s, %s, %s)""",
            (user["id"], refresh_token, datetime.now(timezone.utc) + timedelta(days=7)),
        )
        conn.commit()

    return {"access_token": access_token, "refresh_token": refresh_token, "user": user}, None


def refresh_access_token(conn, refresh_token_str):
    try:
        payload = decode_token(refresh_token_str)
        if payload.get("type") != "refresh":
            return None, "Invalid token type"
    except Exception:
        return None, "Invalid refresh token"

    user_id = payload.get("sub")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM refresh_tokens WHERE user_id = %s AND token = %s AND expires_at > NOW()",
            (user_id, refresh_token_str),
        )
        if not cur.fetchone():
            return None, "Refresh token not found or expired"

        cur.execute(
            "SELECT id, email, role FROM users WHERE id = %s AND is_active = true",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None, "User not found"

    access_token = create_access_token({"sub": row[0], "role": row[2]})
    return {"access_token": access_token, "token_type": "bearer"}, None


def logout_user(conn, user_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM refresh_tokens WHERE user_id = %s", (user_id,))
        conn.commit()


def create_password_reset_token(conn, email):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        if not row:
            return None
        token = generate_token()
        cur.execute(
            """INSERT INTO password_reset_tokens (user_id, token, expires_at)
               VALUES (%s, %s, %s)""",
            (row[0], token, datetime.now(timezone.utc) + timedelta(hours=1)),
        )
        conn.commit()
        return token


def reset_password(conn, token, new_password):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM password_reset_tokens WHERE token = %s AND expires_at > NOW()",
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return False, "Invalid or expired token"
        pw_hash = hash_password(new_password)
        cur.execute("UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s", (pw_hash, row[0]))
        cur.execute("DELETE FROM password_reset_tokens WHERE token = %s", (token,))
        conn.commit()
        return True, None


def verify_email(conn, token):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM email_verification_tokens WHERE token = %s AND expires_at > NOW()",
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return False
        cur.execute("UPDATE users SET is_verified = true, updated_at = NOW() WHERE id = %s", (row[0],))
        cur.execute("DELETE FROM email_verification_tokens WHERE token = %s", (token,))
        conn.commit()
        return True
