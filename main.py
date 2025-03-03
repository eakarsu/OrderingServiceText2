import base64
import json
import os
import time

import assemblyai as aai
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv
import uvicorn
from orderChat import orderChat
#from orderChat import OrderSystem
import ngrok
import asyncio
import google.cloud.texttospeech as tts
import requests
import re


# Load environment variables
load_dotenv()

# Initialize AssemblyAI and Twilio API settings
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
TWILIO_SAMPLE_RATE = 8000  # Hz
# chatService = None

# FastAPI setup
app = FastAPI()
PORT = 5003

# Twilio authentication and ngrok setup
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
api_key = os.getenv('TWILIO_API_KEY_SID')
api_secret = os.getenv('TWILIO_API_SECRET')
client = Client(api_key, api_secret, account_sid)
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
WHATSAPP_NUMBER = os.getenv('WHATSAPP_NUMBER')

print (f" grok: {os.getenv('GROK_AUTHTOKEN')}")
ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))
INCOMING_CALL_ROUTE = '/incoming-call'
INCOMING_TEXT_ROUTE = '/sms'

# Twilio language and voice settings
TWILIO_LANG_CODE = 'en-US'
TWILIO_VOICE = 'Polly.Kimberly-Neural'
# Get from the restaurant table, not hard code
FORWARDING_NUMBER = '+19175587915'
ORIGIN = os.getenv('STORE_ADDRESS')
CALLER_ID = None


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_service_tts_key.json"
tts_client = tts.TextToSpeechClient()

# Dictionary to hold session data
data_sessions = {}
chat_sessions = {}


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return {"message": "Convergenie Food Ordering System is running!"}


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


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def voice(request: Request):
    """Twilio webhook to handle incoming calls and questions."""
    not_initial_call = request.query_params.get('not_initial_call', 'false').lower() == 'true'
    global CALLER_ID
    # Parse the form data from the incoming Twilio webhook request
    if request.method == "POST":
        form_data = await request.form()  # Parse form data for POST request
        CALLER_ID = form_data.get("From")  # Get the caller ID
    elif request.method == "GET":
        query_params = request.query_params  # Parse query params for GET request
        CALLER_ID = query_params.get("From")  # Get the caller ID
    if CALLER_ID:
        print(f"DEBUG: Incoming call received from {CALLER_ID}.")
    else:
        CALLER_ID = "UNKNOWN"  # Default if 'From' is not present
        print("DEBUG: Caller ID could not be retrieved.")

    response = VoiceResponse()
    if not_initial_call:
        # Dial human agent
        # Dial the forwarding number
        response.dial(FORWARDING_NUMBER)
    else:
        # find a way to skip this step and have chatbot start directly
        # response.say("Falls Pizza! How can I help you?", voice="man")
        host = request.url.hostname
        connect = Connect()
        connect.stream(url=f'wss://{host}/media-stream')
        response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def transcription_websocket(websocket: WebSocket):
    await websocket.accept()
    global chatService
    chatService = orderChat(CALLER_ID)

    try:
        # Start the conversation as soon as the websocket connection is established
        while True:
            try:
                data = json.loads(await websocket.receive_text())
            except:
                return
            match data['event']:
                case "connected":
                    print('Transcriber connected')
                case "start":
                    stream_sid = data['start']['streamSid']
                    transcriber = TwilioTranscriber2(websocket, stream_sid)
                    transcriber.connect()
                    print('Twilio started')
                    greeting_message = "Hello! Welcome to Falls Pizza. How can I assist you today?"
                    # Synthesize the greeting message and send as audio to the customer
                    await send_text_as_audio(greeting_message, websocket, stream_sid)
                case "media":
                    payload_b64 = data['media']['payload']
                    payload_mulaw = base64.b64decode(payload_b64)
                    # stream input audio to transcriber.
                    # It will transcribe, then get LLM response and send back to Twilio as text.
                    transcriber.stream(payload_mulaw)
                case "stop":
                    print('Twilio stopped')
                    await transcriber.cleanup()
                    transcriber.close()
                    print('Transcriber closed')
    except:
        print("Websocket closed")
        await websocket.close()
        return
    await websocket.close()


async def send_text_as_audio(text, ws, stream_sid):
    """Convert text to speech and send as audio via WebSocket."""
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(
        language_code="en-IN",
        name="en-IN-Standard-C",
        ssml_gender=tts.SsmlVoiceGender.MALE,
    )
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    audio_content = response.audio_content
    # Send the audio in chunks to the WebSocket
    chunk_size = 320
    for i in range(0, len(audio_content), chunk_size):
        chunk = audio_content[i:i + chunk_size]
        b64_audio_chunk = base64.b64encode(chunk).decode('utf-8')
        json_message = json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": b64_audio_chunk
            }
        })
        await ws.send_text(json_message)
        await asyncio.sleep(0.02)  # Simulate real-time streaming with a delay



class TwilioTranscriber2(aai.RealtimeTranscriber):
    def __init__(self, ws, sid):
        self.ws = ws
        self.sid = sid
        self.loop = asyncio.get_event_loop()  # Get the main event loop
        self.sending_audio = False  # Flag to track audio sending state
        super().__init__(
            on_data=self.on_data_wrapper,
            on_error=on_error,
            on_open=on_open, # optional
            on_close=on_close, # optional
            sample_rate=TWILIO_SAMPLE_RATE,
            encoding=aai.AudioEncoding.pcm_mulaw,
            word_boost=["eddys", "iskender", "doner"],
        )
        self.tasks = []

    def on_data_wrapper(self, data):
        # Schedule on_data to run on the main event loop
        self.loop.call_soon_threadsafe(self.schedule_on_data, data)


    def schedule_on_data(self, data):
        # Create and schedule the async task in the main loop context
        task = asyncio.create_task(on_data(data, self.ws, self.sid, self))
        self.tasks.append(task)

    async def cleanup(self):
        """Waits for all tasks to finish before closing."""
        await asyncio.gather(*self.tasks)  # Await all tasks


def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID:", session_opened.session_id)


async def on_data(transcript: aai.RealtimeTranscript, ws: WebSocket, stream_sid, transcriber: TwilioTranscriber2):
    if not transcript.text or transcriber.sending_audio:
        return
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        chatResponse = chatService.chatAway(transcript.text)

        # process chatResponse if there is a json message, process accordingly
        chatResponseType = chatService.processChatResponse(chatResponse)
        if chatResponseType == 'CONTINUE':
            transcriber.sending_audio = True  # Set the flag when starting to send audio
            chatResponse = str(chatResponse).replace("*","").replace("-","")
            synthesis_input = tts.SynthesisInput(text=str(chatResponse))
            voice = tts.VoiceSelectionParams(
                language_code="en-IN",
                name="en-IN-Standard-C",
                ssml_gender=tts.SsmlVoiceGender.MALE,
            )
            audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MULAW,
                # audio_encoding=tts.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000,
                # speaking_rate=1.2
            )
            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_content = response.audio_content

            # Define chunk size and delay between chunks
            chunk_size = 320  # 20 ms of audio at 8 kHz, 16-bit (16 bits * 8000 Hz * 20 ms / 8 bits per byte)
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i:i + chunk_size]
                # Encode chunk in base64 as required by Twilio
                b64_audio_chunk = base64.b64encode(chunk).decode('utf-8')
                # Prepare the JSON message with base64 audio payload
                json_message = json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": b64_audio_chunk
                    }
                })
                # Send the JSON message over WebSocket
                await ws.send_text(json_message)

                # Simulate real-time streaming with a delay (20 ms for each chunk)
                await asyncio.sleep(0.02)
            # sleep length of stream??
            time.sleep(1)
            transcriber.sending_audio = False  # Reset the flag after sending audio
        elif chatResponseType == 'FORWARD':
            await call_forward()
        elif chatResponseType == 'DELIVERY_CHECK':
            try:
                adress_str = str(chatResponse).replace('\'', '\"')
                chatResponse_json = chatService.extract_json(adress_str)
                address_json = json.loads(chatResponse_json)
                delivery_address = address_json['address']
                await is_delivery_available(destination=delivery_address, web_socket=ws, max_distance_mile=10)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} with chatResponse: {adress_str}")
            except Exception as e:
                print(f"Unexpected error: {e} with chatResponse: {adress_str}")
        elif chatResponseType == 'EXIT':
            # Extract ETA from a table or POS system
            greeting_message = "Your order will be ready in around 45 minutes. Enjoy your meal! Goodbye."
            # Synthesize the greeting message and send as audio to the customer
            await send_text_as_audio(greeting_message, ws, stream_sid)
            return
        else:
            return
    else:
        print(transcript.text, end="\r")


def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)


def on_close():
    print("Closing Session")


async def call_forward():
    print("DEBUG: Running call_forward()")
    call = client.calls.create(
        to=FORWARDING_NUMBER,
        from_=TWILIO_NUMBER,
        # url=f"{ngrok.get_tunnel().public_url}/incoming-call"
        url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}"
    )
    print(f"Call initiated with SID: {call.sid}")
    return call.sid


async def is_delivery_available(destination, web_socket=None, max_distance_mile=10):
    """
    Checks if delivery is available between two addresses based on a maximum distance.

    :param origin: The starting address (e.g., "123 Main St, New York, NY").
    :param destination: The destination address (e.g., "456 Elm St, Boston, MA").
    :param max_distance_km: Maximum delivery distance in kilometers.
    :return: Tuple (delivery_available: bool, distance: float).
    """
    # Replace with your Google Maps API key
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    # URL for the Google Maps Distance Matrix API
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    # API request parameters
    params = {
        "origins": ORIGIN,
        "destinations": destination,
        "units": "imperial",
        "key": API_KEY
    }

    # Send the request
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200 or "rows" not in data:
        raise Exception("Error fetching data from Google Maps API: " + data.get("error_message", "Unknown error"))

    # Extract distance in kilometers
    distance_info = data["rows"][0]["elements"][0]
    if distance_info["status"] != "OK":
        raise Exception(f"Unable to calculate distance: {distance_info['status']}")

    distance_mile = distance_info["distance"]["value"] / 1000  # Convert meters to kilometers

    print("DEBUG: Distance is {} miles".format(distance_mile))
    # Check if delivery is available
    delivery_available = distance_mile <= max_distance_mile
    # if delivery_available:
    #     await web_socket.send_text("Tell the customer that delivery to the address is available and complete the order.")
    # else:
    #     await web_socket.send_text("Tell the customer that delivery to the address is not available and get a new address or cancel the order.")
    return delivery_available, distance_mile


# class TwilioTranscriber2(aai.RealtimeTranscriber):
#     def __init__(self):
#         super().__init__(
#             on_data=on_data,
#             on_error=on_error,
#             on_open=on_open,
#             on_close=on_close,
#             sample_rate=TWILIO_SAMPLE_RATE,
#             encoding=aai.AudioEncoding.pcm_mulaw,
#             word_boost=["ayran", "iskender", "doner"],
#             # boost_param="high",
#             # speech_model=aai.SpeechModel.best,
#         )


if __name__ == "__main__":
    # Open ngrok tunnel
    listener = ngrok.forward(f"http://localhost:{PORT}")
    print(f"Ngrok tunnel opened at {listener.url()} for port {PORT}")
    NGROK_URL = listener.url()

    # Set webhook for the Twilio number
    twilio_numbers = client.incoming_phone_numbers.list()
    twilio_number_sid = [num.sid for num in twilio_numbers if num.phone_number == TWILIO_NUMBER][0]
    # client.incoming_phone_numbers(twilio_number_sid).update(account_sid, voice_url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}")
    client.incoming_phone_numbers(twilio_number_sid).update(account_sid, sms_url=f"{NGROK_URL}{INCOMING_TEXT_ROUTE}")
    # Do not know how to set WhatsApp webhook


    uvicorn.run(app, host="0.0.0.0", port=PORT)
