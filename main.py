import base64
import json
import os
import time

import assemblyai as aai
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException
from twilio.twiml.voice_response import VoiceResponse, Connect, Start
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
import sys, logging, io
from vosk import Model, KaldiRecognizer
from websockets.exceptions import ConnectionClosed
import websockets
from fastapi.websockets import WebSocketState
import aiohttp
import torchaudio.functional as F
import torch
import speech_recognition as sr
import numpy as np
import torchaudio
import audioop
import librosa
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
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Adam voice

tts_client = tts.TextToSpeechClient()

# Dictionary to hold session data
data_sessions = {}
chat_sessions = {}
active_calls = {}

# At the beginning of your file
try:
    model = Model("vosk-model-en-us-0.22")
    print(f"Vosk model loaded: {model}")
except Exception as e:
    print(f"ERROR loading Vosk model: {str(e)}")
    model = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

# --- G.711 μ-law to 16-bit Linear PCM Lookup Table ---
# Standard table used for decoding G.711 μ-law
_ulaw2lin = np.array([
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
     -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
     -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
     -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
     -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
     -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
     -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
      -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
      -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
      -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
      -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
      -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
       -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
     32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
     23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
     15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
     11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
      7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
      5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
      3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
      2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
      1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
      1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
       876,    844,    812,    780,    748,    716,    684,    652,
       620,    588,    556,    524,    492,    460,    428,    396, # Corrected potential typo from common online tables
       372,    356,    340,    324,    308,    292,    276,    260,
       244,    228,    212,    196,    180,    164,    148,    132,
       120,    112,   104,     96,     88,     80,     72,     64,
        56,     48,     40,     32,     24,     16,      8,      0
], dtype=np.int16)
# ------------------------------------------------------

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

@app.post("/voice")
async def handle_incoming_call_test(request: Request):
    try:
        # Parse form data
        form_data = await request.form()
        caller_id = form_data.get("From")
        call_sid = form_data.get("CallSid")
        
        # Initialize chat service
        active_calls[call_sid] = {
            "caller_id": caller_id,
            "chat_service": orderChat(caller_id)
        }
        
        # Generate Twilio TwiML response
        response = VoiceResponse()
        response.say("Welcome to Melville Deli. What would you like to order today?")
        # Use Connect instead of Start
        connect = Connect()
        connect.stream(url=f"wss://{request.base_url.hostname}/media-stream")
        
        # Add a timeout to keep the connection open longer
        #connect = Connect(action="/call-ended")
        #connect.stream(url=f"wss://{request.base_url.hostname}/media-stream")
        response.append(connect)
        
        # Play greeting AFTER setting up the stream
        
        
        return HTMLResponse(content=str(response), media_type="application/xml")
    
    except Exception as e:
        logging.error(f"Error handling voice call: {str(e)}")
        response = VoiceResponse()
        response.say("We're sorry, but there was an error processing your call.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

@app.post("/call-ended")
async def call_ended(request: Request):
    """Handle call ended event"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    
    if call_sid and call_sid in active_calls:
        del active_calls[call_sid]
    
    response = VoiceResponse()
    return HTMLResponse(content=str(response), media_type="application/xml")



@app.post("/voice")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio voice calls"""
    try:
        # Parse form data instead of JSON (Twilio uses form data)
        form_data = await request.form()
        caller_id = form_data.get("From")
        call_sid = form_data.get("CallSid")
        
        logging.info(f"Incoming call from {caller_id} with SID {call_sid}")
        
        # Initialize chat service for this caller
        from orderChat import orderChat
        active_calls[call_sid] = {
            "caller_id": caller_id,
            "chat_service": orderChat(caller_id)
        }
        
        # Generate Twilio TwiML response to connect WebSocket
        # Note: Twilio uses <Start> and <Stream> elements
        response = VoiceResponse()
        # Add initial greeting via <Say>
        response.say("Welcome to Melville Deli. What would you like to order today?")
        response.play("https://api.twilio.com/cowbell.mp3", loop=0)  # Loop until next TwiML
        connect = Connect()
        connect.stream(url=f"wss://{request.base_url.hostname}/media-stream")
        response.append(connect)
        
        
        return HTMLResponse(content=str(response), media_type="application/xml")
    
    except Exception as e:
        logging.error(f"Error handling voice call: {str(e)}")
        response = VoiceResponse()
        response.say("We're sorry, but there was an error processing your call.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

def validate_media_message(message: dict) -> bool:
    required = ["event", "streamSid", "media"]
    if not all(key in message for key in required):
        print(f"Missing required fields: {required}")
        return False
    if "payload" not in message["media"]:
        print("Missing payload in media object")
        return False
    return True


def ulaw_to_pcm2(audio_bytes):
    """Alternative µ-law to PCM conversion using pydub."""
    if not audio_bytes: return bytes()
    try:
        # Save original for comparison
        with open("debug_input.ulaw", "ab") as f:
            f.write(audio_bytes)
        
        # Use pydub for conversion
        from pydub import AudioSegment
        import io
        
        # Create in-memory file-like object with µ-law data
        ulaw_io = io.BytesIO(audio_bytes)
        
        # Load as µ-law audio
        audio = AudioSegment.from_file(
            ulaw_io, 
            format="mulaw", 
            frame_rate=8000,
            channels=1,
            sample_width=1
        )
        
        # Amplify by 24dB
        audio = audio + 24
        
        # Export as PCM
        pcm_io = io.BytesIO()
        audio.export(pcm_io, format="raw")
        pcm_bytes = pcm_io.getvalue()
        
        # Save debug file
        with open("debug_output_alt.pcm", "ab") as f:
            f.write(pcm_bytes)
            
        return pcm_bytes
    except Exception as e:
        print(f"ERROR in ulaw_to_pcm_alt: {str(e)}")
        return bytes()


def ulaw_to_pcm(audio_bytes):
    """Convert G.711 μ-law to 16-bit PCM with amplification and proper diagnostics."""
    if not audio_bytes: return bytes()
    try:
        # Save input (WRITE MODE instead of APPEND)
        with open("debug_input.ulaw", "ab") as f:  # Changed from "ab" to "wb"
            f.write(audio_bytes)
            
        # Convert μ-law to 16-bit PCM
        pcm_bytes = audioop.ulaw2lin(audio_bytes, 2)
        
        # AMPLIFY THE AUDIO
        amplification_factor = 24  # Increased from 16 to 24 for better recognition
        pcm_bytes = audioop.mul(pcm_bytes, 2, amplification_factor)
        print(f"Applied {amplification_factor}x amplification")
        
        # Add RMS level check
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean(pcm_array.astype(np.float32)**2))
        print(f"Audio RMS level: {rms:.2f}")
        # Only process audio if RMS indicates speech is likely present
        if rms > 5000:  # Set threshold based on your log analysis
            print(f"Audio level good (RMS: {rms:.2f}), processing")
            transcription = transcribe_audio(pcm_bytes).strip()
        else:
            print(f"Audio level too low (RMS: {rms:.2f}), skipping")
            transcription = ""

        # Add detailed range analysis
        if len(pcm_array) > 0:
            max_val = np.max(np.abs(pcm_array))
            print(f"Max amplitude: {max_val} ({max_val/32767.0*100:.1f}% of max)")
            
        # Save output (WRITE MODE instead of APPEND)
        with open("debug_output.pcm", "ab") as f:  # Changed from "ab" to "wb"
            f.write(pcm_bytes)
            
        return pcm_bytes
    except Exception as e:
        print(f"ERROR in ulaw_to_pcm: {str(e)}")
        return bytes()

def resample_for_vosk(pcm_audio_bytes):
    """Properly resample 8kHz PCM audio to 16kHz for Vosk processing."""
    try:
        # Convert bytes to numpy array
        pcm_array = np.frombuffer(pcm_audio_bytes, dtype=np.int16)
        
        # Convert to float32 for processing (-1.0 to 1.0 range)
        float_audio = pcm_array.astype(np.float32) / 32768.0
        
        # Resample from 8kHz to 16kHz
        resampled = librosa.resample(float_audio, orig_sr=8000, target_sr=16000)
        
        # Convert back to 16-bit PCM
        resampled_pcm = (resampled * 32767).astype(np.int16).tobytes()
        
        print(f"Resampled 8kHz audio ({len(pcm_audio_bytes)} bytes) to 16kHz ({len(resampled_pcm)} bytes)")
        return resampled_pcm
        
    except Exception as e:
        print(f"ERROR resampling audio: {e}")
        # Return empty bytes on error instead of original to avoid feeding bad data
        return bytes()


def ulaw_to_pcm2(audio_bytes):
    """Convert μ-law encoded audio to PCM."""
    # Convert bytes to numpy array of uint8
    ulaw_audio = np.frombuffer(audio_bytes, dtype=np.uint8)
    # Convert to float tensor
    tensor_audio = torch.tensor(ulaw_audio, dtype=torch.float32)
    # Decode μ-law to PCM
    pcm_audio = F.mu_law_decoding(tensor_audio, quantization_channels=256)

    with open("debug_input.ulaw", "wb") as f:
        f.write(audio_bytes)
    with open("debug_output.pcm", "wb") as f:
        f.write(pcm_audio)

    # Convert to 16-bit PCM bytes
    return pcm_audio.numpy().astype(np.int16).tobytes()

async def process_user_input(text, websocket, call_id):
    chat_service = active_calls[call_id]["chat_service"]
    print(f"Processing: {text}")
    # Your existing processing logic here
    response_text = chat_service.chatAway(text)
    await send_tts_audio(response_text, websocket, call_id)


async def process_and_respond(transcription, websocket, call_id, stream_sid):
    """Handles the actual RAG/LLM and TTS sending."""
    try:
        chat_service = active_calls[call_id]["chat_service"]
        print(f"Processing transcribed text: {transcription}")
        if not chat_service: # Ensure chat_service exists
             print("ERROR: chat_service not available in process_and_respond")
             return

        # --- Potentially Long Call ---
        response_text = chat_service.chatAway(transcription)
        # --------------------------

        print(f"Generated response: {response_text}")

        # --- Send TTS ---
        # Pass stream_sid explicitly if needed by send_tts_audio
        await send_tts_audio(response_text, websocket, stream_sid) # Pass stream_sid
        print(f"Finished calling send_tts_audio for call {call_id}")

    except Exception as e:
        print(f"ERROR during process_and_respond task: {str(e)}")
        import traceback
        traceback.print_exc()

async def handle_response(transcription, websocket, call_id, stream_sid):
    """
    Launches the processing task and sends initial confirmation/mark.
    """
    if not transcription: return

    # --- Optional: Send an immediate mark or quick TTS "Okay, thinking..." ---
    try:
        thinking_mark = json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": "processing_llm"}})
        await websocket.send_text(thinking_mark)
        # Or even a very quick TTS:
        # await send_tts_audio("Okay, one moment...", websocket, call_id, stream_sid)
        print("Sent processing mark/ack.")
    except Exception as e:
        print(f"Error sending initial mark/ack: {e}")
    # ----------------------------------------------------------------------

    # Launch the heavy lifting as a background task
    asyncio.create_task(process_and_respond(transcription, websocket, call_id, stream_sid))
    print(f"Launched background task process_and_respond for call {call_id}")

def transcribe_audio(pcm_audio_bytes):
    """Fixed Google STT implementation that properly formats audio data."""
    if not pcm_audio_bytes:
        print("WARNING: Empty audio buffer sent to transcribe_audio")
        return ""
    
    try:
        # 1. Convert 16-bit PCM to proper WAV format with header
        import io
        import wave
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(8000)  # 8kHz sample rate
            wav_file.writeframes(pcm_audio_bytes)
        
        wav_data = wav_buffer.getvalue()
        
        # 2. Create AudioData from the properly formatted WAV
        recognizer = sr.Recognizer()
        with io.BytesIO(wav_data) as wav_buffer:
            with sr.AudioFile(wav_buffer) as source:
                audio_data = recognizer.record(source)
        
        # 3. Use Google STT with proper parameters
        print("Sending properly formatted audio to Google API...")
        text = recognizer.recognize_google(
            audio_data,
            language="en-US"
        )
        print(f"Google API returned: '{text}'")
        return text
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except Exception as e:
        print(f"Unexpected error in transcribe_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def freq_modify(pcm_audio_bytes):
    """Resample 8kHz PCM audio to 16kHz for Vosk processing."""
    try:
        # Convert bytes to float array
        y_8k = np.frombuffer(pcm_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample from 8kHz to 16kHz
        import librosa
        y_16k = librosa.resample(y_8k, orig_sr=8000, target_sr=16000)
        
        # Convert back to 16-bit PCM bytes
        audio_16k = (y_16k * 32767).astype(np.int16).tobytes()
        
        return audio_16k
    except Exception as e:
        print(f"Error resampling audio: {e}")
        return pcm_audio_bytes  # Return original as fallback


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()  # <<< Use bytearray for mutable buffer
    last_audio_time = time.time()
    call_id = None
    stream_sid = None
    recognizer = None # Initialize recognizer variable

    try:
        # 1. Handle Initial Connection and Start Event
        print("INFO: connection open")
        initial_message = await websocket.receive_text()
        initial_data = json.loads(initial_message)
        print(f"Initial WebSocket message: {initial_message}")

        if initial_data.get("event") == "connected":
            print("Received connection handshake, waiting for stream start...")
            start_message = await websocket.receive_text()
            start_data = json.loads(start_message)
            print(f"Second WebSocket message: {start_message}")

            if start_data.get("event") == "start":
                stream_sid = start_data["start"]["streamSid"]
                call_id = start_data["start"]["callSid"]
                print(f"Using callSid: {call_id} with streamSid: {stream_sid}")
                print(f"Call ID: {call_id}, Stream SID: {stream_sid}")

                # Store stream_sid associated with call_id if needed elsewhere
                if call_id in active_calls:
                     active_calls[call_id]["stream_sid"] = stream_sid
                else:
                     # Handle case where call_id might not be in active_calls yet
                     # This might happen if /voice endpoint hasn't finished processing
                     # For now, we proceed, but this could be a potential race condition
                     print(f"Warning: call_id {call_id} not found in active_calls during start event.")
                     # Create a basic entry if needed
                     active_calls[call_id] = {"stream_sid": stream_sid, "chat_service": None} # Adjust as needed

                # Initialize Vosk Recognizer HERE (after getting sample rate info if needed)
                # *** Use the NARROWBAND model path ***
                try:
                    #model_path = "vosk-model-en-us-0.22" # <--- CHANGE TO NARROWBAND MODEL PATH
                    model_path = "vosk-model-small-en-us-0.15"
                    if not os.path.exists(model_path):
                         raise FileNotFoundError(f"Vosk model not found at {model_path}")
                    local_model = Model(model_path)
                    recognizer = KaldiRecognizer(local_model, 8000) # Use 8000Hz
                    recognizer.SetWords(True)
                    print("Vosk Recognizer initialized with 8000 Hz")
                except Exception as e:
                    print(f"ERROR initializing Vosk recognizer: {str(e)}")
                    recognizer = None # Ensure recognizer is None if init fails

            else:
                print("ERROR: Expected 'start' event after 'connected', but got:", start_data.get("event"))
                await websocket.close()
                return
        else:
            print("ERROR: Expected 'connected' event first, but got:", initial_data.get("event"))
            await websocket.close()
            return

        # Check if recognizer initialized successfully
        if not recognizer:
             print("ERROR: Recognizer could not be initialized. Closing connection.")
             await websocket.close()
             return

        # 2. Main Processing Loop
        while True:
            try:
                # Use timeout to periodically check for silence
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.2) # Short timeout
                data = json.loads(message)
                event = data.get("event")

                if event == "media":
                    if "media" not in data or "payload" not in data["media"]:
                        print("ERROR: Invalid media event format")
                        continue

                    # Decode and convert audio
                    audio_payload = base64.b64decode(data["media"]["payload"])
                    print(f"Received audio payload: {len(audio_payload)} bytes, first 10 bytes: {audio_payload[:10].hex()}")
                    pcm_audio = ulaw_to_pcm(audio_payload) # Your corrected function returning bytes
                    if not pcm_audio: continue # Skip if conversion failed

                    # Extend the MUTABLE buffer
                    audio_buffer.extend(pcm_audio)
                    last_audio_time = time.time() # Update time on receiving audio
                    # print(f"Buffer size: {len(audio_buffer)} bytes") # Optional debug
                    
                    # In the main processing loop where you process audio:
                    if len(audio_buffer) >= 3200: # REDUCED BUFFER SIZE (0.25 sec instead of 0.5 sec)
                        print(f"Processing buffer due to SIZE: {len(audio_buffer)} bytes")
                        audio_to_process = bytes(audio_buffer)
                        audio_buffer.clear()
                            
                        # Skip Vosk entirely
                        print("Using Google STT directly with fixed formatter...")
                        transcription = transcribe_audio(audio_to_process).strip()
                        
                        # Process valid transcription
                        if transcription:
                            print(f"Transcribed: {transcription}")
                            await handle_response(transcription, websocket, call_id, stream_sid)

                elif event == "dtmf":
                     digit = data["dtmf"].get("digit")
                     print(f"DTMF received: {digit}")
                     # If '#' is pressed, process any remaining audio immediately
                     if digit == "#" and len(audio_buffer) > 0:
                         print(f"Processing buffer due to DTMF #: {len(audio_buffer)} bytes")
                         audio_to_process = bytes(audio_buffer)
                         audio_buffer.clear() # <<< CORRECT: Reset buffer AFTER deciding to process

                         transcription = ""
                         # Try Vosk (use FinalResult might be better here)
                         recognizer.AcceptWaveform(freq_modify(audio_to_process)) # Feed remaining
                         result = json.loads(recognizer.FinalResult()) # Get final result
                         transcription = result.get("text", "").strip()

                         # Fallback to Google
                         if not transcription:
                              print("Vosk returned no final text on DTMF, falling back to Google...")
                              transcription = transcribe_audio(audio_to_process).strip()

                         if transcription:
                              print(f"Transcribed (DTMF): {transcription}")
                              if call_id:
                                   await handle_response(transcription, websocket, call_id,stream_sid)
                              else:
                                   print("ERROR: call_id is None, cannot handle response.")
                         else:
                              print("No transcription available (DTMF trigger).")

                elif event == "stop":
                    print("Call ended by stop event.")
                    # Process any final lingering audio
                    if len(audio_buffer) > 0:
                         print(f"Processing final buffer on STOP: {len(audio_buffer)} bytes")
                         audio_to_process = bytes(audio_buffer)
                         audio_buffer.clear() # Reset buffer
                         # --- Optional: Call transcription one last time ---
                         # ... (similar transcription logic) ...
                    break # Exit loop

                elif event == "ping":
                    await websocket.send_text(json.dumps({"event": "pong"}))

                # else: # Don't print for every media event now
                #     print(f"Unhandled event type: {event}")


            except asyncio.TimeoutError:
                # This block runs if websocket.receive_text() times out (e.g., after 0.2s of no messages)
                # Check for silence timeout HERE
                current_time = time.time()
                if len(audio_buffer) > 0 and (current_time - last_audio_time > 0.5): # 0.5 seconds of silence
                    print(f"Processing buffer due to TIMEOUT ({current_time - last_audio_time:.2f}s silence)")
                    audio_to_process = bytes(audio_buffer) # Copy for processing
                    audio_buffer.clear() # <<< CORRECT: Reset buffer AFTER deciding to process

                    transcription = ""
                    # Try Vosk (use FinalResult here)
                    recognizer.AcceptWaveform(freq_modify(audio_to_process)) # Feed remaining
                    result = json.loads(recognizer.FinalResult()) # Get final result
                    transcription = result.get("text", "").strip()

                    # Fallback to Google
                    if not transcription:
                        print("Vosk returned no final text on timeout, falling back to Google...")
                        transcription = transcribe_audio(audio_to_process).strip()

                    if transcription:
                        print(f"Transcribed (timeout): {transcription}")
                        if call_id:
                             await handle_response(transcription, websocket, call_id,stream_sid)
                        else:
                             print("ERROR: call_id is None, cannot handle response.")
                    else:
                        print("No transcription available (timeout trigger).")
                continue # Continue loop after timeout check

            except websockets.exceptions.ConnectionClosedOK:
                print("WebSocket closed normally.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"WebSocket closed with error: {e.code} - {e.reason}")
                break
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")
                continue
            except Exception as e:
                print(f"ERROR in WebSocket loop: {str(e)}")
                import traceback
                traceback.print_exc()
                break

    finally:
        print(f"Closing WebSocket for call_id: {call_id}")
        if call_id and call_id in active_calls:
            # Perform any necessary cleanup for this call
            if 'recognizer' in active_calls[call_id]:
                # If you stored the recognizer per call, clean it up
                pass
            del active_calls[call_id]
            print(f"Removed call {call_id} from active_calls.")
        # Ensure websocket is closed gracefully if not already
        if websocket.client_state != websockets.protocol.State.CLOSED:
             try:
                 await websocket.close(code=1000)
             except Exception as close_err:
                 # Log the specific error during closure
                 print(f"Error during WebSocket closure: {str(close_err)}")


# Add this to your WebSocket handler
async def keep_alive(websocket):
    """Send periodic pings to keep the connection alive."""
    try:
        while True:
            await asyncio.sleep(15)  # Send ping every 15 seconds
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({"event": "ping"}))
    except Exception as e:
        print(f"Heartbeat error: {str(e)}")


async def send_tts_audio(text, websocket, stream_sid):
    """Convert text to speech using ElevenLabs and send over WebSocket."""
    if not text:
        print("send_tts_audio: Received empty text, nothing to send.")
        return
    if websocket.client_state != websockets.protocol.State.OPEN:
         print("send_tts_audio: WebSocket is not open, cannot send TTS.")
         return

    # --- Retrieve the correct stream_sid ---
    if not stream_sid:
        print(f"ERROR in send_tts_audio: Could not find stream_sid for call_id {stream_sid}")
        return
    # -------------------------------------

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=ulaw_8000"
    headers = {
        "Accept": "audio/mpeg", # Corrected? Or should be audio/basic for ulaw? Check ElevenLabs docs. Usually ulaw is audio/basic or audio/mulaw.
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1", # Or your preferred model
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": True
        }
    }

    print(f"Requesting TTS from ElevenLabs for stream {stream_sid}: '{text[:50]}...'")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(tts_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    print(f"ElevenLabs response status OK ({response.status})")
                    audio_content = await response.read() # Read the raw audio bytes
                    print(f"Received {len(audio_content)} bytes of audio from ElevenLabs.")

                    if not audio_content:
                        print("ERROR: ElevenLabs returned empty audio content.")
                        return

                    chunk_size = 160 * 20 # Send larger chunks (e.g., 400ms) to reduce overhead? Twilio expects 20ms chunks though. Let's stick to 160 for now.
                    chunk_size = 160 # Keep 20ms chunks

                    # --- Send Mark message (Optional but good practice) ---
                    try:
                        mark_message = json.dumps({
                            "event": "mark",
                            "streamSid": stream_sid,
                            "mark": {
                                "name": "tts_playing" # Mark the start of TTS
                            }
                        })
                        # print(f"Sending TTS Start Mark for {stream_sid}")
                        await websocket.send_text(mark_message)
                    except Exception as mark_err:
                        print(f"Error sending TTS start mark: {mark_err}")
                    # ------------------------------------------------------

                    sent_chunks = 0
                    for i in range(0, len(audio_content), chunk_size):
                        chunk = audio_content[i:i + chunk_size]
                        b64_chunk = base64.b64encode(chunk).decode('utf-8')

                        media_message = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": b64_chunk
                            }
                        }

                        # Ensure WebSocket is still open before sending
                        if websocket.client_state != websockets.protocol.State.OPEN:
                             print(f"WebSocket closed before sending chunk {sent_chunks+1}. Aborting TTS.")
                             break

                        try:
                            await websocket.send_text(json.dumps(media_message))
                            sent_chunks += 1
                            # print(f"Sent TTS chunk {sent_chunks} ({len(chunk)} bytes) for {stream_sid}") # Verbose
                            await asyncio.sleep(0.018) # Sleep slightly less than 20ms to try and stay ahead
                        except websockets.exceptions.ConnectionClosed:
                             print(f"Connection closed while sending TTS chunk {sent_chunks+1}")
                             break
                        except Exception as send_err:
                             print(f"Error sending TTS chunk {sent_chunks+1}: {send_err}")
                             break # Stop sending if an error occurs

                    print(f"Finished sending {sent_chunks} TTS chunks for {stream_sid}.")

                    # --- Send Mark message (Optional but good practice) ---
                    try:
                         mark_end_message = json.dumps({
                             "event": "mark",
                             "streamSid": stream_sid,
                             "mark": {
                                 "name": "tts_finished" # Mark the end of TTS
                             }
                         })
                         # print(f"Sending TTS End Mark for {stream_sid}")
                         await websocket.send_text(mark_end_message)
                    except Exception as mark_err:
                         print(f"Error sending TTS end mark: {mark_err}")
                    # ------------------------------------------------------

                else:
                    error_body = await response.text()
                    print(f"ERROR from ElevenLabs: Status {response.status} - Body: {error_body}")

    except aiohttp.ClientError as http_err:
        print(f"HTTP Error connecting to ElevenLabs: {http_err}")
    except websockets.exceptions.ConnectionClosed:
         print("WebSocket closed during TTS request/processing.")
    except Exception as e:
        print(f"Exception during TTS processing/sending: {str(e)}")
        import traceback
        traceback.print_exc()



async def send_tts_audio2(text, websocket, call_id):
    """Convert text to speech using ElevenLabs and send over WebSocket."""
    # At the beginning of send_tts_audio function:
    print(f"Sending TTS audio for text: '{text}' to call_sid: {call_id}")

    text = text.replace("*", "").replace("-", "")
    
    # Request μ-law format at 8000Hz - optimal for Twilio
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=ulaw_8000"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.7
        },
        "model_id": "eleven_monolingual_v1"  # Specify a model
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"Successfully generated audio in μ-law format, length: {len(response.content)} bytes")
            audio_content = response.content
            
            # After receiving the audio content:
            print(f"Audio format details: First few bytes: {audio_content[:20]}")

            # μ-law encoding uses 1 byte per sample
            # At 8000Hz, 160 bytes = 20ms of audio (optimal for Twilio)
            chunk_size = 160
            
            for i in range(0, len(audio_content), chunk_size):
                # After each chunk is sent:
                print(f"Sent audio chunk {i}/{len(audio_content)//chunk_size}")

                chunk = audio_content[i:i + chunk_size]
                b64_chunk = base64.b64encode(chunk).decode('utf-8')
                   
                # Ensure WebSocket is still connected before sending
                try:
                     # Build media message
                    media_message = {
                        "event": "media",
                        "streamSid": active_calls[call_id]["stream_sid"],  # Must match Twilio's streamSid
                        "media": {
                            "payload": b64_chunk
                        }
                    }
                    
                    # Validate and send
                    if validate_media_message(media_message):
                        #print(f"Sending audio chunk {i//chunk_size}")
                        await websocket.send_text(json.dumps(media_message))
                        #await asyncio.sleep(0.02)  # Match 20ms timing
                    else:
                        print("Invalid media message, skipping chunk")
                             
                    # Note: 160 samples at 8000Hz = 20ms, so sleep for that duration
                    #await asyncio.sleep(0.02)
                except Exception as e:
                    print(f"Error sending audio chunk: {str(e)}")
                    break
        else:
            print(f"Error from ElevenLabs: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception during TTS processing: {str(e)}")


def processChatResponse(response_text):
    """Process chat response to determine the response type."""
    try:
        # Check if the response is a JSON string
        if '{' in response_text and '}' in response_text:
            # Try to extract JSON from the response
            json_str = extract_json(response_text)
            if json_str:
                response_data = json.loads(json_str)
                
                # Check for message_type or type
                if 'message_type' in response_data and response_data['message_type'] == 'call_forward':
                    return 'FORWARD'
                elif 'type' in response_data and response_data['type'] == 'is_delivery_available':
                    return 'DELIVERY_CHECK'
                elif 'message_type' in response_data and response_data['message_type'] == 'order':
                    # Process the order
                    #self.processOrder(json.dumps(response_data))
                    return 'EXIT'
        
        # If no special handling required, continue normal conversation
        return 'CONTINUE'
    except Exception as e:
        print(f"Error processing chat response: {str(e)}")
        return 'CONTINUE'

def extract_json(text):
    """Extract JSON string from text."""
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > 0:
            json_str = text[start_idx:end_idx]
            # Validate it's proper JSON
            json.loads(json_str)
            return json_str
        return None
    except Exception as e:
        print(f"Error extracting JSON: {str(e)}")
        return None


def update_twilio_webhook(ngrok_url):
    from twilio.rest import Client
    
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
            
        number = numbers[0]
        updated_number = number.update(
            voice_url=f"{ngrok_url}/voice"
        )
        
        print(f"Successfully updated voice webhook URL to {ngrok_url}/voice")
        return True
    except Exception as e:
        print(f"Failed to update webhook URL: {str(e)}")
        return False



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
    update_twilio_webhook(NGROK_URL)

    # Set webhook for the Twilio number
    twilio_numbers = client.incoming_phone_numbers.list()
    twilio_number_sid = [num.sid for num in twilio_numbers if num.phone_number == TWILIO_NUMBER][0]
    # client.incoming_phone_numbers(twilio_number_sid).update(account_sid, voice_url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}")
    client.incoming_phone_numbers(twilio_number_sid).update(account_sid, sms_url=f"{NGROK_URL}{INCOMING_TEXT_ROUTE}")
    # Do not know how to set WhatsApp webhook


    uvicorn.run(app, host="0.0.0.0", 
                port=PORT,
                ws_ping_interval=60,  # Increase from default 20 seconds
                ws_ping_timeout=30 )   # Increase timeout for pong responses)
