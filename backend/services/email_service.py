import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from GmailSender import GmailSender
from backend.config import settings


def send_verification_email(email: str, token: str):
    if not settings.GMAIL_USER or not settings.GMAIL_PASSWORD:
        print(f"[Email] Verification link: {settings.FRONTEND_URL}/verify-email/{token}")
        return
    sender = GmailSender(settings.GMAIL_USER, settings.GMAIL_PASSWORD)
    body = f"Click to verify your email:\n{settings.FRONTEND_URL}/verify-email/{token}"
    sender.send_email(email, "Verify Your Email", body)


def send_password_reset_email(email: str, token: str):
    if not settings.GMAIL_USER or not settings.GMAIL_PASSWORD:
        print(f"[Email] Reset link: {settings.FRONTEND_URL}/reset-password/{token}")
        return
    sender = GmailSender(settings.GMAIL_USER, settings.GMAIL_PASSWORD)
    body = f"Click to reset your password:\n{settings.FRONTEND_URL}/reset-password/{token}"
    sender.send_email(email, "Reset Your Password", body)
