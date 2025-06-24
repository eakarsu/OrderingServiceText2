import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import httpx
import re
from typing import Dict, Optional
import ngrok
import uvicorn
from orderChat import orderChat
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,Response, FileResponse,JSONResponse
from fastapi.exceptions import HTTPException
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()
app = FastAPI()

# Session management using caller_number as key
sessions = {}

# Configuration
ULTRAVOX_API_KEY = os.getenv("ULTRAVOX_API_KEY")
NGROK_PORT = int(os.getenv("NGROK_PORT", "5003"))

# Global variable to store the dynamic ngrok URL
NGROK_URL = None

# Store active Ultravox calls for cleanup
active_calls: Dict[str, str] = {}
# Dictionary to hold session data
data_sessions = {}
chat_sessions = {}


# --- Add CORS Middleware ---
# Define the origins that are allowed to make cross-site requests.
# It's crucial to include the origins from which your Capacitor app will be served.
origins = [
    "https://orderlybite.com",        # Your production frontend (Vercel)
    "https://www.orderlybite.com",    # Your production frontend with www
    
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

    # It's also good practice to allow your backend's own origin if it ever serves a frontend
    "https://api.orderlybite.com" 
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)  
def update_twilio_webhook(ngrok_url, webhook_type):
    """Updates either voice or SMS webhook for a Twilio phone number."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    phone_number = os.getenv("TWILIO_VOICE_NUMBER")
    
    if not all([account_sid, auth_token, phone_number]):
        print("❌ Missing Twilio credentials in environment variables")
        return False
    
    print(f"🔄 Updating Twilio webhook with URL: {ngrok_url}")
    print(f"Using account SID: {account_sid}")
    
    client = Client(account_sid, auth_token)
    
    try:
        numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not numbers:
            print(f"❌ No phone number found matching {phone_number}")
            return False
        
        number_sid = numbers[0].sid
        
        if webhook_type.lower() == 'voice':
            client.incoming_phone_numbers(number_sid).update(
                voice_url=f"{ngrok_url}/voice",
                status_callback=f"{ngrok_url}/status",
                status_callback_method="POST"
            )
            print(f"✅ Updated voice webhook for {phone_number} to {ngrok_url}/voice")
        
        elif webhook_type.lower() == 'sms':
            client.incoming_phone_numbers(number_sid).update(
                sms_url=f"{ngrok_url}/sms",
                sms_method="POST"
            )
            print(f"✅ Updated SMS webhook for {phone_number} to {ngrok_url}/sms")
        
        else:
            print(f"❌ Invalid webhook type: {webhook_type}. Use 'voice' or 'sms'.")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update {webhook_type} webhook URL: {str(e)}")
        return False

@app.api_route("/sms", methods=["GET", "POST"])
async def sms_reply(request: Request):
    global CALLER_ID

    # Parse form data for POST request
    form_data = await request.form()
    message_body = form_data.get("Body", "").strip()
    CALLER_ID = form_data.get("From", "").replace("whatsapp:","")

    if CALLER_ID:
        if CALLER_ID not in data_sessions:
            print(f"DEBUG: Incoming call received from {CALLER_ID}.")
            chat_sessions[CALLER_ID] = orderChat(CALLER_ID)
    else:
        CALLER_ID = "UNKNOWN"  # Default if 'From' is not present
        print("DEBUG: Caller ID could not be retrieved.")

    # Initialize session if not already started
    if CALLER_ID not in data_sessions:
        data_sessions[CALLER_ID] = []

    # Handle exit command
    if message_body.lower() == "exit":
        del data_sessions[CALLER_ID]
        response = MessagingResponse()
        response.message("Session ended. Goodbye!")
        return HTMLResponse(content=str(response), media_type="application/xml")

    # Save the message to the session
    data_sessions[CALLER_ID].append(message_body)

    # Generate a response
    response = MessagingResponse()
    chatResponse = chat_sessions[CALLER_ID].chatAway(message_body)
    response.message(str(chatResponse))
    print("CHATBOT: {}".format(str(chatResponse)))

    print("DEBUG: Send following to caller: {}".format(response.to_xml()))
    return HTMLResponse(content=response.to_xml(), media_type="application/xml")


@app.post("/voice")
async def voice(request: Request):
    """Handle incoming Twilio calls - create Ultravox call with orderChat integration"""
    
    form = await request.form()
    phone_number_called = form.get("To")
    caller_number = form.get("From")
    call_sid = form.get("CallSid")
    
    print(f"📞 Incoming call from {caller_number} to {phone_number_called}, Call SID: {call_sid}")
    
    # ✅ INITIALIZE YOUR EXISTING ORDERCHAT SYSTEM
    if caller_number not in sessions:
        print(f"🍽️ Initializing orderChat for {caller_number}")
        sessions[caller_number] = orderChat(caller_number)  # Uses all your vector DB logic
    
    # Create Ultravox call that will use your orderChat as a tool
    ultravox_call = await create_ultravox_call_with_orderchat(call_sid, caller_number)
    
    if ultravox_call:
        ultravox_call_id = ultravox_call["callId"] 
        active_calls[ultravox_call_id] = call_sid
        
        response = VoiceResponse()
        connect = response.connect()
        connect.stream(url=ultravox_call["joinUrl"])
        return Response(str(response), media_type="application/xml")
    else:
        response = VoiceResponse()
        response.say("Sorry, our system is temporarily unavailable.")
        response.hangup()
        return Response(str(response), media_type="application/xml")


async def create_ultravox_call_with_orderchat(call_sid: str, caller_number: str):
    """Create Ultravox call with your existing orderChat system as a tool"""
    
     # ✅ SIMPLIFIED PROMPT - Remove the confusing instructions
    system_prompt = f"""
    You are a voice assistant for Melville Deli. When a customer speaks:

    1. Call the processOrder tool with their exact words
    2. Speak the response from processOrder directly to the customer
    3. Wait for the customer's next input

    Do not call processOrder with your own responses - only with customer speech.
    Start by calling processOrder with an empty string to get the greeting.
    
    Customer phone: {caller_number}
    """
    
    headers = {
        "X-API-Key": ULTRAVOX_API_KEY,
        "Content-Type": "application/json"
    }
    
    print(f"📡 Creating Ultravox call with orderChat integration for {caller_number}")
    
    payload = {
        "model": "fixie-ai/ultravox-70B",
        "systemPrompt": system_prompt,
        "medium": {"twilio": {}},
        "voice": "b0e6b5c1-3100-44d5-8578-9015aa3023ae",
        "firstSpeaker": "FIRST_SPEAKER_AGENT",
        "initialOutputMedium": "MESSAGE_MEDIUM_VOICE",
        "selectedTools": [
            {
                "temporaryTool": {
                    "modelToolName": "processOrder",
                    "description": "Process customer speech using the restaurant ordering system. Only call this with actual customer speech, never with bot responses.",
                    "dynamicParameters": [
                        {
                            "name": "user_input",
                            "location": "PARAMETER_LOCATION_BODY",
                            "schema": {
                                "description": "Customer's actual speech (empty string for greeting)",
                                "type": "string"
                            },
                            "required": True
                        },
                        {
                            "name": "caller_number",
                            "location": "PARAMETER_LOCATION_BODY", 
                            "schema": {
                                "description": "Customer's phone number",
                                "type": "string"
                            },
                            "required": True
                        }
                    ],
                    "http": {
                        "baseUrlPattern": f"{NGROK_URL}/process-order",
                        "httpMethod": "POST"
                    }
                }
            }
        ],
        "metadata": {
            "twilio_call_sid": call_sid,
            "caller_number": caller_number,
            "restaurant": "Melville Deli"
        }
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.ultravox.ai/api/calls",
                headers=headers,
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 201:
                call_data = response.json()
                print(f"🎙️ Ultravox call created with orderChat integration: {call_data['callId']}")
                print(f"🔗 Join URL: {call_data['joinUrl']}")
                return call_data
            else:
                print(f"❌ Failed to create Ultravox call: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error creating Ultravox call: {e}")
        return None

@app.post("/process-order")
async def process_order(request: Request):
    """Use your existing orderChat system to process customer input"""
    
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        caller_number = data.get("caller_number", "")
        
        print(f"🍽️ Processing order for {caller_number}: '{user_input}'")
        
        # ✅ USE YOUR EXISTING ORDERCHAT SYSTEM
        if caller_number not in sessions:
            print(f"🔄 Creating new orderChat session for {caller_number}")
            sessions[caller_number] = orderChat(caller_number)  # Your existing class with all vector DB logic
        
        # ✅ USE YOUR EXISTING CHATAWAY METHOD WITH ALL YOUR PROMPTS AND RULES
        order_chat = sessions[caller_number]
        response = order_chat.chatAway(user_input)
        
        print(f"🤖 OrderChat response: {response}")
        
        return {"response": response}
        
    except Exception as e:
        print(f"❌ Error processing order: {e}")
        return {"response": "I'm sorry, there was an error processing your request. Could you please repeat that?"}

@app.post("/status")
async def status(request: Request):
    """Handle Twilio status callbacks"""
    form = await request.form()
    call_status = form.get("CallStatus")
    call_sid = form.get("CallSid")
    caller_number = form.get("From")
    
    print(f"📊 Twilio call {call_sid} status: {call_status}")
    
    if call_status in ["completed", "failed", "busy", "no-answer"]:
        # Clean up sessions by caller_number
        if caller_number and caller_number in sessions:
            del sessions[caller_number]
            print(f"🧹 Cleaned up orderChat session for {caller_number}")
        
        # Also clean up Ultravox mapping
        ultravox_call_to_remove = None
        for uv_call_id, tw_call_sid in active_calls.items():
            if tw_call_sid == call_sid:
                ultravox_call_to_remove = uv_call_id
                break
        
        if ultravox_call_to_remove:
            del active_calls[ultravox_call_to_remove]
            print(f"🧹 Cleaned up Ultravox mapping for call {call_sid}")
    
    return Response(status_code=204)

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "melville-deli-voice-ai-orderchat",
        "active_sessions": len(sessions),
        "active_ultravox_calls": len(active_calls),
        "ngrok_url": NGROK_URL,
        "port": NGROK_PORT
    }

@app.get("/debug/sessions")
async def debug_sessions():
    return {
        "active_orderchat_sessions": list(sessions.keys()),
        "ultravox_calls": active_calls,
        "total_active": len(sessions),
        "ngrok_url": NGROK_URL
    }

@app.get("/debug/test-orderchat/{phone_number}")
async def test_orderchat(phone_number: str, query: str = ""):
    """Test endpoint to verify orderChat is working"""
    try:
        if phone_number not in sessions:
            sessions[phone_number] = orderChat(phone_number)
        
        response = sessions[phone_number].chatAway(query)
        return {
            "phone_number": phone_number,
            "query": query,
            "response": response,
            "session_exists": True
        }
    except Exception as e:
        return {
            "phone_number": phone_number,
            "query": query,
            "error": str(e),
            "session_exists": False
        }

if __name__ == "__main__":
    print("🍔 Starting Melville Deli Voice AI Server with OrderChat Integration...")
    
    # Use amazon_main.py ngrok logic exactly
    authtoken = os.getenv("NGROK_AUTHTOKEN")
    ngrok.set_auth_token(authtoken)
    
    # Open ngrok tunnel
    listener = ngrok.forward(f"http://localhost:{NGROK_PORT}")
    print(f"🚀 Ngrok tunnel opened at {listener.url()} for port {NGROK_PORT}")
    NGROK_URL = listener.url()
    
    # Update Twilio webhooks
    print(f"📡 Updating Twilio webhooks to point to {NGROK_URL}")
    update_twilio_webhook(NGROK_URL, "voice")
    update_twilio_webhook(NGROK_URL, "sms")
    
    # Start the server
    print(f"🎯 Server starting with full orderChat integration...")
    print(f"📞 Voice endpoint: {NGROK_URL}/voice")
    print(f"🍽️ Order processing endpoint: {NGROK_URL}/process-order")
    print(f"🔍 Debug endpoint: {NGROK_URL}/debug/sessions")
    print(f"🧪 Test endpoint: {NGROK_URL}/debug/test-orderchat/+1234567890?query=coffee")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=NGROK_PORT,
        ws_ping_interval=60,
        ws_ping_timeout=30
    )
