import os
import uuid
import boto3
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from orderChat import orderChat  # Your universal bot
from dotenv import load_dotenv
import ngrok
from twilio.rest import Client
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse,Response, FileResponse,JSONResponse
from fastapi.exceptions import HTTPException
import uvicorn
import sys, logging, io
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant
import requests
from fastapi.responses import FileResponse
import json
import re 
from typing import Dict, List, Optional
import asyncio

load_dotenv()  # Loads .env if present
PORT = int(os.getenv('PORT', 5003))  # Default to 5003 if not set
sessions = {}

app = FastAPI()
# --- Add CORS Middleware ---
# Define the origins that are allowed to make cross-site requests.
# It's crucial to include the origins from which your Capacitor app will be served.
origins = [
    "https://orderlybite.com",        # Your production frontend (Vercel)
    "https://www.orderlybite.com",    # Your production frontend with www
    "https://omniassistai.com",        # Your production frontend (Vercel)
    "https://www.omniassistai.com",    # Your production frontend with www
    # Origins for Capacitor apps [6]
    "capacitor://localhost",        # For Capacitor iOS local scheme
    "ionic://localhost",            # Another common Capacitor local scheme (though less used with raw Capacitor)
    "http://localhost",             # For Capacitor Android local scheme AND potentially some iOS WKWebView scenarios if not using a custom scheme

    # Origins for local development servers
    "http://localhost:5173",        # Common Vite dev server port (if you use live reload: ionic cap run ios -l)
    "http://localhost:8100",        # Common Ionic serve port (if you use live reload: ionic cap run ios -l)
    
    # Your previously listed local dev server ports
    "http://localhost:8080",        
    "http://localhost:5003",        
    "http://localhost:5006",  
    "http://localhost:5007",

    # It's also good practice to allow your backend's own origin if it ever serves a frontend
    "https://api.orderlybite.com",
    "https://api.omniassistai.com" 
]
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)  

polly = boto3.client("polly", region_name="us-east-1")
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
authtoken = os.getenv("NGROK_AUTHTOKEN")
ngrok.set_auth_token(authtoken)


# Dictionary to hold session data
data_sessions = {}
chat_sessions = {}
active_calls = {}

def update_twilio_webhook(ngrok_url, webhook_type):
    """
    Updates either voice or SMS webhook for a Twilio phone number.
    
    Args:
        ngrok_url (str): The base ngrok URL
        webhook_type (str): Either 'voice' or 'sms'
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Get credentials from environment variables
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    phone_number = os.getenv("TWILIO_VOICE_NUMBER")
    
    print(f"Using account SID: {account_sid}")
    
    # Initialize Twilio client
    client = Client(account_sid, auth_token)
    
    try:
        # Find the phone number
        numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not numbers:
            print(f"No phone number found matching {phone_number}")
            return False
        
        number_sid = numbers[0].sid
        
        # Update based on webhook type
        if webhook_type.lower() == 'voice':
            client.incoming_phone_numbers(number_sid).update(
                voice_url=f"{ngrok_url}/voice",
                status_callback=f"{ngrok_url}/status",
                status_callback_method="POST"
            )
            print(f"Updated voice webhook for {phone_number} to {ngrok_url}/voice")
        
        elif webhook_type.lower() == 'sms':
            client.incoming_phone_numbers(number_sid).update(
                sms_url=f"{ngrok_url}/sms",
                sms_method="POST"
            )
            print(f"Updated SMS webhook for {phone_number} to {ngrok_url}/sms")
        
        else:
            print(f"Invalid webhook type: {webhook_type}. Use 'voice' or 'sms'.")
            return False
        
        return True
        
    except Exception as e:
        print(f"Failed to update {webhook_type} webhook URL: {str(e)}")
        return False


@app.post("/status")
async def status(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    print(f"Status callback: CallSid={call_sid}, CallStatus={call_status}")

    # Clean up session when call is completed
    if call_status == "completed" and call_sid in sessions:
        sessions.pop(call_sid, None)
        print(f"Cleaned up session for CallSid={call_sid}")
    return Response(status_code=204)
  

@app.api_route("/sms", methods=["GET", "POST"])
async def sms_reply(request: Request):
    global CALLER_ID
    
    print(f"🔍 SMS endpoint called - Method: {request.method}")
    
    # Handle both form data and JSON data
    try:
        # Try form data first (for Twilio webhooks and curl)
        form_data = await request.form()
        message_body = form_data.get("Body", "").strip()
        CALLER_ID = form_data.get("From", "").replace("whatsapp:", "")
        print(f"✅ Form data - Body: '{message_body}', From: '{CALLER_ID}'")
    except Exception as form_error:
        print(f"⚠️ Form parsing failed, trying JSON: {form_error}")
        # Fallback to JSON data (for frontend requests)
        try:
            json_data = await request.json()
            message_body = json_data.get("message", "").strip()
            CALLER_ID = json_data.get("from", "frontend_user")
            print(f"✅ JSON data - message: '{message_body}', from: '{CALLER_ID}'")
        except Exception as json_error:
            print(f"❌ Both form and JSON parsing failed: {json_error}")
            message_body = ""
            CALLER_ID = "frontend_user"
    
    # Ensure CALLER_ID is not empty or "UNKNOWN"
    if not CALLER_ID or CALLER_ID == "UNKNOWN":
        old_caller_id = CALLER_ID
        CALLER_ID = "frontend_user"
        print(f"🔄 Changed CALLER_ID from '{old_caller_id}' to '{CALLER_ID}'")
    
    # Always ensure session exists
    if CALLER_ID not in chat_sessions:
        print(f"🔍 Creating new session for '{CALLER_ID}'...")
        try:
            chat_sessions[CALLER_ID] = orderChat(CALLER_ID)
            print(f"✅ Session created for '{CALLER_ID}'")
        except Exception as e:
            print(f"❌ Failed to create session for '{CALLER_ID}': {e}")
            response = MessagingResponse()
            response.message("I'm here to help with your request. Could you tell me more about what you need?")
            return HTMLResponse(content=str(response), media_type="application/xml")

    # Initialize data session if not already started
    if CALLER_ID not in data_sessions:
        data_sessions[CALLER_ID] = []

    # Handle exit command
    if message_body.lower() == "exit":
        if CALLER_ID in data_sessions:
            del data_sessions[CALLER_ID]
        if CALLER_ID in chat_sessions:
            del chat_sessions[CALLER_ID]
        response = MessagingResponse()
        response.message("Session ended. Goodbye!")
        return HTMLResponse(content=str(response), media_type="application/xml")

    # Save the message to the session
    data_sessions[CALLER_ID].append(message_body)

    # Generate a response with error handling
    response = MessagingResponse()
    try:
        print(f"🔍 Processing message for '{CALLER_ID}': '{message_body}'")
        chatResponse = chat_sessions[CALLER_ID].chatAway(message_body)
        response.message(str(chatResponse))
        print("CHATBOT: {}".format(str(chatResponse)))
    except Exception as e:
        print(f"❌ chatAway error for '{CALLER_ID}': {e}")
        response.message("I'm here to help with your request. Could you tell me more about what you need?")

    print("DEBUG: Send following to caller: {}".format(response.to_xml()))
    return HTMLResponse(content=response.to_xml(), media_type="application/xml")
   

@app.post("/voice")
async def voice(request: Request):
    response = VoiceResponse()
    connect = response.connect()
    # Use your public WebSocket URL here (wss://...)
    clean_url = re.sub(r'^https?://', '', NGROK_URL)
    connect.conversation_relay(
        url=f"wss://{clean_url}/ws",
        welcome_greeting="Welcome to Melville Deli! Please tell me your order."
    )
    return Response(str(response), media_type="application/xml")


async def stream_chataway_response(response_text: str):
    """Stream chatAway response like LLM streaming - character by character"""
    # Stream character by character (mimicking LLM token streaming)
    for i, char in enumerate(response_text):
        if char:  # Only yield non-empty characters
            yield {
                "token": char,
                "last": False,
                "type": "text",
            }
            # Small delay to simulate natural streaming pace
            await asyncio.sleep(0.05)
    
    # Final empty token (matching your draft_response pattern)
    yield {
        "token": "",
        "type": "text",
        "last": True,
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    call_sid: Optional[str] = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print("ConvRelay Message: ", message)

            if message["type"] == "setup":
                call_sid = message["callSid"]
                print(f"Setup initiated for call {call_sid} from **{message.get('from', 'unknown')[-4:]}")
                
                # Initialize your orderChat session (instead of conversation history)
                sessions[str(call_sid)] = orderChat(call_sid)

            elif message["type"] == "prompt":
                user_text = message["voicePrompt"]
                print("User said: ", user_text)

                if call_sid and str(call_sid) in sessions:
                    # Use your chatAway method instead of LLM
                    order_chat = sessions[str(call_sid)]
                    
                    # Get complete response from chatAway
                    chataway_response = order_chat.chatAway(user_text)
                    print("ChatAway response: ", chataway_response)
                    
                    # Stream the response like LLM streaming
                    async for event in stream_chataway_response(chataway_response):
                        await websocket.send_json(event)

            elif message["type"] == "interrupt":
                print("Response interrupted")

            elif message["type"] == "error":
                print("Error:", message)

            else:
                print("Unknown message type:", message.get("type"))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if call_sid and str(call_sid) in sessions:
            sessions.pop(str(call_sid), None)
        print("Client has disconnected.")


@app.post('/token')
async def token():
    # Your environment variables are correct
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    api_key = os.getenv('TWILIO_API_KEY_SID')
    api_secret = os.getenv('TWILIO_API_SECRET')
    twiml_app_sid = os.getenv('TWIML_APP_SID')
    
    # Create access token with voice grant
    access_token = AccessToken(account_sid, api_key, api_secret, identity='user123')
    
    # Create voice grant
    voice_grant = VoiceGrant(
        outgoing_application_sid=twiml_app_sid,
        incoming_allow=True
    )
    
    access_token.add_grant(voice_grant)
    
    # Use FastAPI's JSONResponse instead of Flask's jsonify
    return JSONResponse(content={'token': access_token.to_jwt()})

@app.post('/voice-status')
async def voice_status(request: Request):
    form_data = await request.form()
    call_status = form_data.get("CallStatus")
    print(f"Voice call status: {call_status}")
    return JSONResponse(content={'msg':"OK"})

if __name__ == "__main__":

    # Open ngrok tunnel
    listener = ngrok.forward(f"http://localhost:{PORT}")
    print(f"Ngrok tunnel opened at {listener.url()} for port {PORT}")
    NGROK_URL = listener.url()
    update_twilio_webhook(NGROK_URL, "voice")
    update_twilio_webhook(NGROK_URL, "sms")

    uvicorn.run(app, host="0.0.0.0", 
                port=PORT,
                ws_ping_interval=60,  # Increase from default 20 seconds
                ws_ping_timeout=30 ) 
    

