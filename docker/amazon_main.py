import os
import uuid
import boto3
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from orderChat import orderChat  # Your conversational logic class
from dotenv import load_dotenv
import ngrok
from twilio.rest import Client
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse,Response, FileResponse
from fastapi.exceptions import HTTPException
import uvicorn
import sys, logging, io

load_dotenv()  # Loads .env if present
PORT = 5003
sessions = {}

app = FastAPI()
polly = boto3.client("polly", region_name="us-east-1")
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
authtoken = os.getenv("NGROK_AUTHTOKEN")
ngrok.set_auth_token(authtoken)


def update_twilio_webhook(ngrok_url):

    # Get credentials from environment variables
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    # Use the auth token shown in your Twilio console (with the "Show" button)
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")  # Not API_SECRET
    phone_number = os.getenv("TWILIO_VOICE_NUMBER") # Hardcode for testing
    
    print(f"Using account SID: {account_sid}")
    
    # Initialize Twilio client
    client = Client(account_sid, auth_token)
    
    try:
        # Update the voice URL for your phone number
        numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not numbers:
            print(f"No phone number found matching {phone_number}")
            return False
            
        number_sid = numbers[0].sid
        client.incoming_phone_numbers(number_sid).update(voice_url=f"{ngrok_url}/voice")
        client.incoming_phone_numbers(number_sid).update(
                voice_url=f"{ngrok_url}/voice",
                status_callback=f"{ngrok_url}/status",
                status_callback_method="POST",
                status_callback_event=["completed"]
            )
        
        print(f"Updated webhook for {phone_number} to {ngrok_url}/voice")
        return True
        
        print(f"Successfully updated voice webhook URL to {ngrok_url}/voice")
        return True
    except Exception as e:
        print(f"Failed to update webhook URL: {str(e)}")
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
  
# --- 3. Polly synthesis ---
def synthesize_with_polly(text, voice_id="Joanna"):
    polly_response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine="neural"
    )
    audio_filename = f"/tmp/{uuid.uuid4()}.mp3"
    with open(audio_filename, "wb") as f:
        f.write(polly_response["AudioStream"].read())
    return audio_filename

# --- 4. Flask routes ---
@app.post("/voice")
async def voice(request: Request):
    try:
        form = await request.form()
        phone_number_called = form.get("To")
        transcription = form.get("SpeechResult", "")
        
        call_sid = form.get("CallSid") or phone_number_called

        if call_sid not in sessions:
            sessions[call_sid] = orderChat(phone_number_called)
            sessions[call_sid].prompt_count = 0
        py = sessions[call_sid]
        py.prompt_count += 1

        print(f"Call to: {phone_number_called}, User said: {transcription} call_sid:{call_sid}")
        response = VoiceResponse()
        if not transcription:
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=5,
                barge_in=True   # Enable barge-in!
            )
            if py.prompt_count == 1:
                # Initial prompt
                gather.say("Welcome Melville Deli! Please tell me your order.", voice="Polly.Joanna-Neural",language="en-US")
            else:
                # Repeat prompt or different message
                gather.say("Sorry, I didn't hear anything. Please tell me your order.", voice="Polly.Joanna-Neural",language="en-US")

            response.append(gather)
            response.redirect("/voice")
            return Response(str(response), media_type="application/xml")
    
       
        # 2. Call chatService with transcription
        response_text = py.chatAway(transcription)
        print("Bot response:", response_text)

        # 5. Optionally, gather next user input for multi-turn dialog
        # Next turn: respond and gather more input
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
        return Response(content=str(response), media_type="application/xml")


if __name__ == "__main__":

    # Open ngrok tunnel
    listener = ngrok.forward(f"http://localhost:{PORT}")
    print(f"Ngrok tunnel opened at {listener.url()} for port {PORT}")
    NGROK_URL = listener.url()
    update_twilio_webhook(NGROK_URL)

    uvicorn.run(app, host="0.0.0.0", 
                port=PORT,
                ws_ping_interval=60,  # Increase from default 20 seconds
                ws_ping_timeout=30 ) 
    

