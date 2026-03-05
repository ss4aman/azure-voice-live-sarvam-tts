"""Handles media streaming to Azure Voice Live API via WebSocket."""

import asyncio
import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from azure.identity.aio import ManagedIdentityCredential
from websockets.asyncio.client import connect as ws_connect
from websockets.typing import Data

from .ambient_mixer import AmbientMixer
from .sarvam_tts import SarvamTTS

logger = logging.getLogger(__name__)

# Default chunk size in bytes (100ms of audio at 24kHz, 16-bit mono)
DEFAULT_CHUNK_SIZE = 4800  # 24000 samples/sec * 0.1 sec * 2 bytes


def _load_puri_bank_mock_db() -> dict:
    """Load mock bank data from JSON file."""
    candidate_paths = [
        Path(__file__).resolve().parents[1] / "data" / "puri_bank_mock_accounts.json",
        Path(__file__).resolve().parents[2] / "app" / "data" / "puri_bank_mock_accounts.json",
        Path.cwd() / "app" / "data" / "puri_bank_mock_accounts.json",
    ]
    configured = os.getenv("PURI_BANK_DATA_FILE", "").strip()
    if configured:
        candidate_paths.insert(0, Path(configured))

    for data_file in candidate_paths:
        try:
            if data_file.exists():
                with data_file.open("r", encoding="utf-8") as f:
                    bank_data = json.load(f)
                logger.info(
                    "Loaded Puri Bank mock data from %s with %s accounts",
                    str(data_file),
                    len(bank_data.get("accounts", [])),
                )
                return bank_data
        except Exception as err:
            logger.warning("Failed reading mock data file %s: %s", str(data_file), err)

    logger.error("Puri Bank mock data file not found. Checked: %s", [str(p) for p in candidate_paths])
    return {"bankName": "Puri Bank", "currency": "INR", "accounts": []}


def _build_puri_bank_instructions() -> str:
    """Build system instructions with Puri Bank persona and mock data."""
    bank_data = _load_puri_bank_mock_db()
    bank_name = bank_data.get("bankName", "Puri Bank")
    currency = bank_data.get("currency", "INR")
    accounts = bank_data.get("accounts", [])

    account_lines = []
    for acct in accounts:
        loan = acct.get('loan', {})
        if loan.get('active'):
            loan_info = (
                f"  *** LOAN ACTIVE = YES ***\n"
                f"  *** LOAN EMI = {currency} {loan.get('emiAmount', 0)} ***\n"
                f"  *** LOAN NEXT DUE DATE = {loan.get('nextDueDate', 'N/A')} ***"
            )
        else:
            loan_info = "  *** LOAN ACTIVE = NO (no loan on this account) ***"
        account_lines.append(
            f"ACCOUNT {acct.get('accountId')}:\n"
            f"  customerName={acct.get('customerName')}\n"
            f"  mobileLast4={acct.get('registeredMobileLast4')}\n"
            f"  dobDayMonth={acct.get('dobDayMonth')}\n"
            f"  accountType={acct.get('accountType')}\n"
            f"  *** ACCOUNT BALANCE = {currency} {acct.get('balance')} ***\n"
            f"{loan_info}"
        )
    account_context = "\n\n".join(account_lines)
    full_records_context = json.dumps(accounts, ensure_ascii=False)

    custom_instructions = os.getenv("PURI_BANK_SYSTEM_INSTRUCTIONS", "").strip()

    base_instructions = (
        f"You are a friendly phone banking voice agent for {bank_name}. "
        "Your name is Kavya (काव्या).\n\n"
        "CALL FLOW — follow this EXACT sequence:\n"
        "1. GREETING (first message, word-for-word): "
        "नमस्ते! मैं काव्या बात कर रही हूँ पुरी बैंक से। सबसे पहले आपका वेरिफ़िकेशन कर लेते हैं — आपका अकाउंट आईडी बताइए?\n"
        "2. Complete all 3 verification steps (see VERIFICATION section below).\n"
        "3. After verification passes, say 'जी, वेरिफ़ाई हो गया। बताइए, क्या मदद करूँ?'\n"
        "4. Now help the customer with whatever they ask — NO more verification needed for this account.\n"
        "5. Customer can ask multiple things (balance, EMI, transactions) — answer directly, no re-verification.\n"
        "6. Only re-verify if customer asks about a DIFFERENT account ID.\n\n"
        "GREETING RULES (CRITICAL):\n"
        "- Output EXACTLY the greeting above — same words, same order. Do NOT rephrase or rearrange.\n"
        "- The greeting INCLUDES asking for account ID — this kicks off verification immediately.\n"
        "- NEVER repeat greeting or 'क्या मदद करूँ' until AFTER verification is complete.\n\n"
        "LANGUAGE RULES:\n"
        "- ALWAYS respond in Hindi DEVANAGARI script. NEVER use Latin/Roman script.\n"
        "- NUMBERS: Always spell out in Hindi words — 12,450 → 'बारह हज़ार चार सौ पचास रुपये'. NEVER write digits.\n"
        "- DATES: Spell in Hindi — 28 Feb → 'अट्ठाईस फ़रवरी'. NEVER write digits for dates.\n"
        "- BANKING TERMS in Devanagari: बैलेंस, अकाउंट, लोन, ईएमआई, यूपीआई, लेनदेन\n"
        "- Write abbreviations WITHOUT dots — ईएमआई (NOT ई.एम.आई.), यूपीआई (NOT यू.पी.आई.), "
        "आईडी (NOT आई.डी.), एसएमएस (NOT एस.एम.एस.). Dots between letters cause pronunciation errors.\n"
        "- No decimal points or .0 in amounts. Round to nearest rupee.\n\n"
        "STYLE — INFORMAL FRIENDLY PHONE CONVERSATION:\n"
        "- ALWAYS speak in COMPLETE sentences. Never leave a sentence half-finished.\n"
        "- Reply in MAX 1-2 SHORT but COMPLETE sentences. Be brief like a real phone call.\n"
        "- Sound warm, friendly and caring — like a helpful friend at the bank, NOT formal.\n"
        "- Use natural fillers like 'जी', 'हाँ जी', 'बिल्कुल', 'अच्छा', 'ज़रूर', 'हाँ हाँ'.\n"
        "- Use casual, easy-going desi conversational Hindi — no formal/bookish/शुद्ध हिंदी language.\n"
        "- You are female: use 'मैं बताती हूँ', 'मैं देखती हूँ' etc.\n"
        "- NEVER repeat yourself. If you already said something, don't say it again.\n"
        "- NEVER volunteer or reveal account details the customer didn't ask for. "
        "Only share the SPECIFIC information the customer requested.\n\n"
        "FAREWELL RULES:\n"
        "- NEVER say 'आपका स्वागत है' — this is a welcome phrase, NOT a goodbye.\n"
        "- When customer says they don't need anything or wants to end the call, say something casual like: "
        "'अच्छा जी, कोई बात नहीं! कभी भी ज़रूरत हो तो बताइएगा।' or 'ठीक है जी, अपना ख्याल रखियेगा।'\n"
        "- Keep goodbyes warm and casual, never formal.\n\n"
        "🔒 VERIFICATION — ONE-TIME SECURITY GATE:\n"
        "Verification happens ONCE at the start of the call. After passing, the customer is trusted for the rest of the call.\n\n"
        "WHAT IS BLOCKED until verification is complete:\n"
        "- Balance, transactions, loan details, EMI, account type, ANY account detail\n"
        "- Even confirming whether an account exists or a loan is active\n\n"
        "THE 3 VERIFICATION STEPS — IN ORDER, ONE PER TURN:\n"
        "STEP 1: (Already asked in greeting) Verify account ID against database.\n"
        "STEP 2: Ask 'अकाउंट से जुड़े मोबाइल नंबर के आखिरी चार डिजिट बताइए?' → wait → verify.\n"
        "STEP 3: Ask 'और बस एक बात — आपकी जन्मतिथि बता दीजिए?' → wait → verify.\n\n"
        "VERIFICATION RULES:\n"
        "- Ask ONE step per turn. NEVER combine two questions.\n"
        "- NEVER skip any step. All 3 must pass.\n"
        "- If customer gets impatient, say: 'जी, बस सिक्योरिटी के लिए है। एक मिनट में हो जाएगा।'\n"
        "- If ANY step fails twice, say: 'माफ़ कीजिए, वेरिफ़िकेशन नहीं हो पा रहा। मैं आपको हमारे एजेंट से जोड़ती हूँ।' STOP.\n"
        "- After ALL THREE pass → say 'जी, वेरिफ़ाई हो गया। बताइए, क्या मदद करूँ?'\n\n"
        "AFTER VERIFICATION:\n"
        "- Customer is now VERIFIED for this account. Do NOT ask for verification again.\n"
        "- Answer all questions about this account directly — balance, EMI, transactions, loan — whatever they ask.\n"
        "- Do NOT re-verify for every question. They are already verified.\n"
        "- ONLY re-verify if customer asks about a DIFFERENT account ID.\n\n"
        "DATA RULES:\n"
        "- Use ONLY the database below. Don't invent data.\n"
        "- ONLY share what the customer SPECIFICALLY asked for. Do NOT volunteer anything extra.\n"
        "- All amounts in Hindi words.\n"
        "- CRITICAL: 'ACCOUNT BALANCE' and 'LOAN EMI' are TWO COMPLETELY DIFFERENT fields in the database. "
        "Read the EXACT number from the correct field. "
        "For PB1001: BALANCE=1,28,450 and EMI=12,450. These are NOT the same. "
        "If customer asks for balance, say the BALANCE number. If customer asks for EMI, say the EMI number. "
        "Double-check which field you are reading before answering.\n"
        "- LOAN LOOKUP: Check the 'LOAN ACTIVE' field for the account. "
        "PB1001 has LOAN ACTIVE = YES (EMI 12,450, due 28 March). "
        "PB1002 has LOAN ACTIVE = NO. "
        "PB1003 has LOAN ACTIVE = YES (EMI 28,300, due 28 March). "
        "If customer asks about loan/EMI, check LOAN ACTIVE first. If YES, share EMI amount and due date. "
        "If NO, say there is no active loan. NEVER say 'no loan' for an account that has LOAN ACTIVE = YES.\n\n"
        "ACCOUNT ID MATCHING:\n"
        "- Customer may say account ID in Hindi — 'पीबी एक हज़ार एक' or 'पी बी 1001' all mean PB1001.\n"
        "- If transcription gives 'टीबी' or 'पीवी' etc., treat it as 'पीबी' (PB) — these are common speech recognition errors.\n"
        "- Be flexible with account ID matching: ignore spaces, case, and minor transcription errors.\n\n"
        "EXAMPLE (for PB1001):\n"
        "Balance query: 'जी, आपका बैलेंस एक लाख अट्ठाईस हज़ार चार सौ पचास रुपये है। और कुछ बताऊँ?'\n"
        "EMI query: 'जी, आपकी लोन ईएमआई बारह हज़ार चार सौ पचास रुपये है, अगली तारीख़ अठ्ठाईस मार्च है।'\n\n"
        f"Database ({bank_name}):\n{account_context}\n\n"
        f"Full records:\n{full_records_context}\n"
    )

    if custom_instructions:
        return f"{base_instructions}\nAdditional instructions:\n{custom_instructions}"
    return base_instructions


def session_config():
    """Returns the default session configuration for Voice Live."""
    instructions = _build_puri_bank_instructions()
    return {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": instructions,
            "turn_detection": {
                "type": "azure_semantic_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "remove_filler_words": True,
                "end_of_utterance_detection": {
                    "model": "semantic_detection_v1",
                    "threshold": 0.1,
                    "timeout": 1.5,
                },
            },
            "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
            "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
            "input_audio_transcription": {
                "model": "azure-speech",
                "language": "hi-IN",
            },
            # Voice config omitted: modalities=["text"] means Voice Live
            # does not generate audio. Sarvam TTS is used for speech synthesis.
        },
    }


class ACSMediaHandler:
    """Manages audio streaming between client and Azure Voice Live API."""

    def __init__(self, config):
        self.endpoint = config["AZURE_VOICE_LIVE_ENDPOINT"]
        self.model = config["VOICE_LIVE_MODEL"]
        self.api_key = config["AZURE_VOICE_LIVE_API_KEY"]
        self.client_id = config["AZURE_USER_ASSIGNED_IDENTITY_CLIENT_ID"]
        self.send_queue = asyncio.Queue()
        self.ws = None
        self.send_task = None
        self.incoming_websocket = None
        self.is_raw_audio = True

        # TTS output buffering for continuous ambient mixing
        self._tts_output_buffer = bytearray()
        self._tts_buffer_lock = asyncio.Lock()
        self._max_buffer_size = 480000  # 10 seconds of audio - large enough for long responses
        self._buffer_warning_logged = False
        self._tts_playback_started = False  # Track if we've started playing TTS
        self._min_buffer_to_start = 9600  # 200ms buffer before starting TTS playback
        
        # Sarvam TTS initialization
        sarvam_api_key = config.get("SARVAM_API_KEY", "")
        sarvam_speaker = config.get("SARVAM_SPEAKER", "kavya")
        sarvam_language = config.get("SARVAM_TARGET_LANGUAGE", "hi-IN")
        sarvam_pace = config.get("SARVAM_PACE", 1.1)
        sarvam_temperature = config.get("SARVAM_TEMPERATURE", 0.7)
        if sarvam_api_key:
            self._sarvam_tts = SarvamTTS(
                api_key=sarvam_api_key,
                speaker=sarvam_speaker,
                target_language=sarvam_language,
                pace=sarvam_pace,
                temperature=sarvam_temperature,
            )
            logger.info("[VoiceLiveACSHandler] Sarvam TTS enabled (speaker=%s, lang=%s, pace=%.2f, temp=%.2f)", sarvam_speaker, sarvam_language, sarvam_pace, sarvam_temperature)
        else:
            self._sarvam_tts = None
            logger.warning("[VoiceLiveACSHandler] SARVAM_API_KEY not set - TTS disabled")

        # Ambient mixer initialization
        self._ambient_mixer: Optional[AmbientMixer] = None
        ambient_preset = config.get("AMBIENT_PRESET", "none")
        if ambient_preset and ambient_preset != "none":
            try:
                self._ambient_mixer = AmbientMixer(preset=ambient_preset)
            except Exception as e:
                logger.error(f"Failed to initialize AmbientMixer: {e}")

    def _generate_guid(self):
        return str(uuid.uuid4())

    async def connect(self):
        """Connects to Azure Voice Live API via WebSocket."""
        endpoint = self.endpoint.rstrip("/")
        model = self.model.strip()
        url = f"{endpoint}/voice-live/realtime?api-version=2025-05-01-preview&model={model}"
        url = url.replace("https://", "wss://")

        headers = {"x-ms-client-request-id": self._generate_guid()}

        if self.client_id:
            # Use async context manager to auto-close the credential
            async with ManagedIdentityCredential(client_id=self.client_id) as credential:
                token = await credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
                headers["Authorization"] = f"Bearer {token.token}"
                logger.info("[VoiceLiveACSHandler] Connected to Voice Live API by managed identity")
        else:
            headers["api-key"] = self.api_key

        self.ws = await ws_connect(url, additional_headers=headers)
        logger.info("[VoiceLiveACSHandler] Connected to Voice Live API")

        await self._send_json(session_config())
        await self._send_json({"type": "response.create"})

        # Set up Sarvam TTS audio callback to stream audio back to client
        if self._sarvam_tts:
            self._sarvam_tts.set_audio_callback(self._on_sarvam_audio)

        asyncio.create_task(self._receiver_loop())
        self.send_task = asyncio.create_task(self._sender_loop())

    async def init_incoming_websocket(self, socket, is_raw_audio=True):
        """Sets up incoming ACS WebSocket."""
        self.incoming_websocket = socket
        self.is_raw_audio = is_raw_audio

    async def audio_to_voicelive(self, audio_b64: str):
        """Queues audio data to be sent to Voice Live API."""
        await self.send_queue.put(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
        )

    async def _send_json(self, obj):
        """Sends a JSON object over WebSocket."""
        if self.ws:
            await self.ws.send(json.dumps(obj))

    async def _sender_loop(self):
        """Continuously sends messages from the queue to the Voice Live WebSocket."""
        try:
            while True:
                msg = await self.send_queue.get()
                if self.ws:
                    await self.ws.send(msg)
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Sender loop error")

    async def _receiver_loop(self):
        """Handles incoming events from the Voice Live WebSocket."""
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")

                match event_type:
                    case "session.created":
                        session_id = event.get("session", {}).get("id")
                        logger.info("[VoiceLiveACSHandler] Session ID: %s", session_id)

                    case "input_audio_buffer.cleared":
                        logger.info("Input Audio Buffer Cleared Message")

                    case "input_audio_buffer.speech_started":
                        logger.info(
                            "Voice activity detection started at %s ms",
                            event.get("audio_start_ms"),
                        )
                        # Clear Sarvam TTS buffer on user interruption
                        if self._sarvam_tts:
                            self._sarvam_tts.clear_buffer()
                        await self.stop_audio()

                    case "input_audio_buffer.speech_stopped":
                        logger.info("Speech stopped")

                    case "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript")
                        logger.info("User: %s", transcript)

                    case "conversation.item.input_audio_transcription.failed":
                        error_msg = event.get("error")
                        logger.warning("Transcription Error: %s", error_msg)

                    case "response.done":
                        response = event.get("response", {})
                        logger.info("Response Done: Id=%s", response.get("id"))
                        if response.get("status_details"):
                            logger.info(
                                "Status Details: %s",
                                json.dumps(response["status_details"], indent=2),
                            )

                    case "response.audio_transcript.done":
                        transcript = event.get("transcript")
                        logger.info("AI (audio transcript): %s", transcript)
                        await self.send_message(
                            json.dumps({"Kind": "Transcription", "Text": transcript})
                        )

                    case "response.text.delta":
                        # Text-only modality: incremental text from LLM
                        text_delta = event.get("delta", "")
                        if text_delta and self._sarvam_tts:
                            await self._sarvam_tts.add_text_delta(text_delta)

                    case "response.text.done":
                        # Text-only modality: full text response complete
                        full_text = event.get("text", "")
                        logger.info("AI: %s", full_text)
                        # Send transcript to client UI
                        await self.send_message(
                            json.dumps({"Kind": "Transcription", "Text": full_text})
                        )
                        # Flush any remaining text through Sarvam TTS
                        if self._sarvam_tts:
                            await self._sarvam_tts.flush_remaining()

                    case "response.audio.delta":
                        # Skip Voice Live audio when Sarvam TTS handles speech
                        if self._sarvam_tts:
                            continue
                        delta = event.get("delta")
                        audio_bytes = base64.b64decode(delta)
                        
                        # Check if ambient mixing is enabled
                        if self._ambient_mixer is not None and self._ambient_mixer.is_enabled():
                            # Buffer TTS for continuous output mixing
                            async with self._tts_buffer_lock:
                                self._tts_output_buffer.extend(audio_bytes)
                                # Warn if buffer is getting large, but NEVER drop audio
                                if len(self._tts_output_buffer) > self._max_buffer_size:
                                    if not self._buffer_warning_logged:
                                        logger.warning(
                                            f"TTS buffer large: {len(self._tts_output_buffer)} bytes. "
                                            "Speech may be delayed but will not be cut."
                                        )
                                        self._buffer_warning_logged = True
                                elif self._buffer_warning_logged and len(self._tts_output_buffer) < self._max_buffer_size // 2:
                                    self._buffer_warning_logged = False  # Reset warning flag
                        else:
                            # No ambient - send immediately (original behavior)
                            if self.is_raw_audio:
                                await self.send_message(audio_bytes)
                            else:
                                await self.voicelive_to_acs(delta)

                    case "error":
                        logger.error("Voice Live Error: %s", event)

                    case _:
                        logger.debug(
                            "[VoiceLiveACSHandler] Other event: %s", event_type
                        )
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Receiver loop error")

    async def send_message(self, message: Data):
        """Sends data back to client WebSocket."""
        try:
            await self.incoming_websocket.send(message)
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Failed to send message")

    async def voicelive_to_acs(self, base64_data):
        """Converts Voice Live audio delta to ACS audio message."""
        try:
            data = {
                "Kind": "AudioData",
                "AudioData": {"Data": base64_data},
                "StopAudio": None,
            }
            await self.send_message(json.dumps(data))
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Error in voicelive_to_acs")

    async def _on_sarvam_audio(self, pcm_bytes: bytes):
        """Callback for Sarvam TTS audio output.
        
        Sends synthesized PCM audio back to the client. For ambient-enabled
        sessions, buffers audio for mixing. Otherwise sends directly.
        """
        try:
            if self._ambient_mixer is not None and self._ambient_mixer.is_enabled():
                # Buffer for ambient mixing
                async with self._tts_buffer_lock:
                    self._tts_output_buffer.extend(pcm_bytes)
                    if len(self._tts_output_buffer) > self._max_buffer_size:
                        if not self._buffer_warning_logged:
                            logger.warning(
                                "TTS buffer large: %d bytes", len(self._tts_output_buffer)
                            )
                            self._buffer_warning_logged = True
            else:
                # No ambient - send audio directly to client
                if self.is_raw_audio:
                    # Web browser - send raw PCM bytes in chunks
                    chunk_size = DEFAULT_CHUNK_SIZE
                    for i in range(0, len(pcm_bytes), chunk_size):
                        chunk = pcm_bytes[i : i + chunk_size]
                        await self.send_message(chunk)
                else:
                    # Phone call (ACS) - send as base64 JSON in chunks
                    chunk_size = DEFAULT_CHUNK_SIZE
                    for i in range(0, len(pcm_bytes), chunk_size):
                        chunk = pcm_bytes[i : i + chunk_size]
                        audio_b64 = base64.b64encode(chunk).decode("ascii")
                        data = {
                            "Kind": "AudioData",
                            "AudioData": {"Data": audio_b64},
                            "StopAudio": None,
                        }
                        await self.send_message(json.dumps(data))
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Error sending Sarvam TTS audio")

    async def stop_audio(self):
        """Sends a StopAudio signal to ACS."""
        stop_audio_data = {"Kind": "StopAudio", "AudioData": None, "StopAudio": {}}
        await self.send_message(json.dumps(stop_audio_data))
        
        # Clear TTS buffer when user starts speaking
        if self._ambient_mixer is not None:
            async with self._tts_buffer_lock:
                self._tts_output_buffer.clear()
                self._tts_playback_started = False

    async def _send_continuous_audio(self, chunk_size: int) -> None:
        """
        Send continuous audio (ambient + TTS if available) back to client.
        
        Called for every incoming audio frame, ensuring continuous output.
        Uses buffered TTS with minimum buffer threshold to prevent mid-word cuts.
        
        Args:
            chunk_size: Size of audio chunk to send (matches incoming frame size)
        """
        if self._ambient_mixer is None or not self._ambient_mixer.is_enabled():
            return  # Ambient disabled, skip
            
        try:
            async with self._tts_buffer_lock:
                buffer_len = len(self._tts_output_buffer)
                
                # Always get a consistent ambient chunk first
                ambient_bytes = self._ambient_mixer.get_ambient_only_chunk(chunk_size)
                
                # Determine if we should play TTS
                should_play_tts = False
                if self._tts_playback_started:
                    # Already playing - continue until buffer empty
                    if buffer_len >= chunk_size:
                        should_play_tts = True
                    elif buffer_len > 0:
                        # Partial buffer but still playing - use what we have
                        should_play_tts = True
                    else:
                        # Buffer empty - stop playback mode
                        self._tts_playback_started = False
                else:
                    # Not yet playing - wait for minimum buffer
                    if buffer_len >= self._min_buffer_to_start:
                        self._tts_playback_started = True
                        should_play_tts = True
                
                if should_play_tts and buffer_len >= chunk_size:
                    # Full TTS chunk available - add TTS on top of ambient
                    tts_chunk = bytes(self._tts_output_buffer[:chunk_size])
                    del self._tts_output_buffer[:chunk_size]
                    
                    # Mix: ambient (constant) + TTS
                    ambient = np.frombuffer(ambient_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    tts = np.frombuffer(tts_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    mixed = ambient + tts
                    mixed = np.clip(mixed, -0.95, 0.95)  # Soft limit
                    output_bytes = (mixed * 32767).astype(np.int16).tobytes()
                    
                elif should_play_tts and buffer_len > 0:
                    # Partial TTS remaining at end of speech - drain it
                    tts_chunk = bytes(self._tts_output_buffer[:])
                    self._tts_output_buffer.clear()
                    self._tts_playback_started = False
                    
                    ambient = np.frombuffer(ambient_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Only mix TTS for the portion we have
                    tts_samples = len(tts_chunk) // 2
                    tts = np.frombuffer(tts_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    ambient[:tts_samples] += tts
                    mixed = np.clip(ambient, -0.95, 0.95)
                    output_bytes = (mixed * 32767).astype(np.int16).tobytes()
                    
                else:
                    # No TTS ready - just send constant ambient
                    output_bytes = ambient_bytes
            
            # Send to client
            if self.is_raw_audio:
                # Web browser - raw bytes
                await self.send_message(output_bytes)
            else:
                # Phone call - JSON wrapped
                output_b64 = base64.b64encode(output_bytes).decode("ascii")
                data = {
                    "Kind": "AudioData",
                    "AudioData": {"Data": output_b64},
                    "StopAudio": None,
                }
                await self.send_message(json.dumps(data))
                
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Error in _send_continuous_audio")

    async def acs_to_voicelive(self, stream_data):
        """Processes audio from ACS and forwards to Voice Live if not silent."""
        try:
            data = json.loads(stream_data)
            if data.get("kind") == "AudioData":
                audio_data = data.get("audioData", {})
                incoming_data = audio_data.get("data", "")
                
                # Determine chunk size from incoming audio
                if incoming_data:
                    incoming_bytes = base64.b64decode(incoming_data)
                    chunk_size = len(incoming_bytes)
                else:
                    chunk_size = DEFAULT_CHUNK_SIZE
                
                # Send continuous audio back to caller (ambient + TTS mixed)
                await self._send_continuous_audio(chunk_size)
                
                # Forward non-silent audio to Voice Live (existing logic)
                if not audio_data.get("silent", True):
                    await self.audio_to_voicelive(audio_data.get("data"))
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Error processing ACS audio")

    async def web_to_voicelive(self, audio_bytes):
        """Encodes raw audio bytes and sends to Voice Live API."""
        chunk_size = len(audio_bytes)
        
        # Send continuous audio back to browser (ambient + TTS mixed)
        await self._send_continuous_audio(chunk_size)
        
        # Forward to Voice Live
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        await self.audio_to_voicelive(audio_b64)

    async def stop_audio_output(self):
        """Clean up resources on disconnect."""
        try:
            if self._sarvam_tts:
                self._sarvam_tts.clear_buffer()
                await self._sarvam_tts.close()
            if self.ws:
                await self.ws.close()
            if self.send_task:
                self.send_task.cancel()
        except Exception:
            logger.exception("[VoiceLiveACSHandler] Error during stop_audio_output")
