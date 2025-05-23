import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from orderChat import orderChat
from dotenv import load_dotenv
import ngrok
import uvicorn
import redis
import pickle
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
PORT = 5003

app = FastAPI()
origins = [
    "https://orderlybite.com",
    "https://www.orderlybite.com"
    # Add localhost for development if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
authtoken = os.getenv("NGROK_AUTHTOKEN")
ngrok.set_auth_token(authtoken)

# --- Sessions dict for context ---
sessions = {}
redis_client = redis.Redis(host='localhost', port=6379, db=0)
def update_twilio_webhook(ngrok_url):
    from twilio.rest import Client
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    phone_number = os.getenv("TWILIO_NUMBER")
    client = Client(account_sid, auth_token)
    try:
        numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not numbers:
            print(f"No phone number found matching {phone_number}")
            return False
        number_sid = numbers[0].sid
        client.incoming_phone_numbers(number_sid).update(voice_url=f"{ngrok_url}/voice")
        print(f"Updated webhook for {phone_number} to {ngrok_url}/voice")
        return True
    except Exception as e:
        print(f"Failed to update webhook URL: {str(e)}")
        return False

@app.post("/voice")
async def voice(request: Request):
    try:
        form = await request.form()
        phone_number_called = form.get("To")
        transcription = form.get("SpeechResult", "")
        call_sid = form.get("CallSid") or phone_number_called

        print(f"Call to: {phone_number_called}, User said: {transcription}")

        # Try to load session from Redis
        session_key = f"orderchat:{call_sid}"
        # When loading session
        if redis_client.exists(session_key):
            state_data = pickle.loads(redis_client.get(session_key))
            py = orderChat(phone_number_called)
            py.set_state(state_data)  # Implement this method to restore state
        else:
            py = orderChat(phone_number_called)

            
        response = VoiceResponse()

        if not transcription:
            # First turn: welcome prompt
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=5,
                barge_in=True
            )
            gather.say("Welcome to Melville Deli. What would you like to order?",voice="Polly.Joanna-Neural",language="en-US")
            response.append(gather)
            response.redirect("/voice")
            return Response(str(response), media_type="application/xml")

        
        response_text = py.chatAway(transcription)
        print("Bot response:", response_text)
        # When saving session
        state_data = py.get_state()  # Implement this method to return a dict of serializable state
        redis_client.set(session_key, pickle.dumps(state_data))

        # Respond and gather more input
        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            timeout=5,
            barge_in=True
        )
        gather.say(response_text,voice="Polly.Joanna-Neural",language="en-US")
        response.append(gather)
        response.redirect("/voice")
        return Response(str(response), media_type="application/xml")

    except Exception as e:
        logging.error(f"Error handling voice call: {str(e)}")
        response = VoiceResponse()
        response.say("We're sorry, but there was an error processing your call.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

if __name__ == "__main__":
    listener = ngrok.forward(f"http://localhost:{PORT}")
    print(f"Ngrok tunnel opened at {listener.url()} for port {PORT}")
    NGROK_URL = listener.url()
    update_twilio_webhook(NGROK_URL)
    print ("Using this")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

