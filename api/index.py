from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your orderChat class (adjust as needed)
from orderChat import orderChat

# Load environment variables
load_dotenv()

# Twilio configuration
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
api_key = os.getenv('TWILIO_API_KEY_SID')
api_secret = os.getenv('TWILIO_API_SECRET')
client = Client(api_key, api_secret, account_sid)
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
WHATSAPP_NUMBER = os.getenv('WHATSAPP_NUMBER')

# Initialize FastAPI app
app = FastAPI()

# Import backend config
from backend.config import settings

# Add CORS middleware
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
from backend.middleware.error_handler import ErrorHandlerMiddleware
from backend.middleware.rate_limiter import RateLimiterMiddleware
from backend.middleware.security_headers import SecurityHeadersMiddleware

app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimiterMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Include admin dashboard routers
from backend.routers import auth, users, orders, categories, menu_items, dashboard

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(orders.router)
app.include_router(categories.router)
app.include_router(menu_items.router)
app.include_router(dashboard.router)

# Global chat sessions dictionary
chat_sessions = {}

@app.get("/")
async def index():
    return {"message": "Food and Salon Ordering API is running!"}

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = data.get('input', '')
        caller_id = data.get('caller_id', '+19175587915')

        # Initialize or retrieve existing chat session
        if caller_id not in chat_sessions:
            chat_sessions[caller_id] = orderChat(caller_id)

        # Process the user input
        response = chat_sessions[caller_id].chatAway(user_input)

        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/sms")
async def sms_reply(request: Request):
    form_data = await request.form()
    message_body = form_data.get("Body", "").strip()
    caller_id = form_data.get("From", "").replace("whatsapp:", "")

    # Initialize chat session if needed
    if caller_id not in chat_sessions:
        chat_sessions[caller_id] = orderChat(caller_id)

    # Process message
    response = MessagingResponse()
    chat_response = chat_sessions[caller_id].chatAway(message_body)
    response.message(str(chat_response))

    return HTMLResponse(content=str(response), media_type="application/xml")

@app.post("/api/reset")
async def reset_session(request: Request):
    try:
        data = await request.json()
        caller_id = data.get('caller_id', '+19175587915')

        if caller_id in chat_sessions:
            del chat_sessions[caller_id]

        return {"status": "success", "message": "Chat session reset successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/categories")
async def get_categories(caller_id: str = '+19175587915'):
    try:
        if caller_id not in chat_sessions:
            chat_sessions[caller_id] = orderChat(caller_id)

        categories = chat_sessions[caller_id].processor.indexer.categories_col.get()

        if categories and "documents" in categories:
            return {"categories": categories["documents"]}
        else:
            return {"categories": []}
    except Exception as e:
        return {"status": "error", "message": str(e)}
