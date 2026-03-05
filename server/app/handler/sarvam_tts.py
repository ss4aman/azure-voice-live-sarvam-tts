"""Sarvam AI Text-to-Speech client for converting text to audio.

Uses a parallel-batch queue with ordered delivery — when multiple
sentences are queued, fires all TTS API calls simultaneously and
delivers audio in the original text order.  For N sentences the API
wait drops from  sum(t1…tN)  to  max(t1…tN).
"""

import asyncio
import base64
import io
import logging
import re
import time
import wave
from typing import Callable, Awaitable, Optional

import httpx

logger = logging.getLogger(__name__)

# Split on sentence-ending punctuation: . ! ? ।
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?।])\s+')

# Minimum text length before flushing to TTS
MIN_SENTENCE_LENGTH = 20


class SarvamTTS:
    """Client for Sarvam AI Text-to-Speech REST API.

    Uses a parallel-batch queue: fires all queued TTS calls at once
    and delivers results in order, reducing multi-sentence latency
    from sum to max.
    """

    def __init__(
        self,
        api_key: str,
        speaker: str = "kavya",
        target_language: str = "hi-IN",
        model: str = "bulbul:v3",
        sample_rate: int = 24000,
        pace: float = 1.35,
        temperature: float = 0.7,
    ):
        self.api_key = api_key
        self.speaker = speaker
        self.target_language = target_language
        self.model = model
        self.sample_rate = sample_rate
        self.pace = pace
        self.temperature = temperature
        self.api_url = "https://api.sarvam.ai/text-to-speech"

        # Persistent HTTP client — reuses TCP+TLS connections across calls,
        # eliminating ~100-300ms handshake overhead per TTS request.
        # Tuned for parallel batch: up to 4 concurrent TTS calls.
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=10.0),
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=4),
        )

        # Text accumulation buffer for sentence-level streaming
        self._text_buffer = ""
        self._audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None

        # Sequential TTS queue — ensures audio chunks are sent in order
        self._tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def set_audio_callback(self, callback: Callable[[bytes], Awaitable[None]]):
        """Set callback function that receives PCM audio bytes."""
        self._audio_callback = callback
        # Start the sequential worker
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._tts_worker())

    async def _tts_worker(self):
        """Parallel-batch worker with ordered delivery.

        Collects ALL currently-queued sentences, fires their TTS API
        calls simultaneously, and delivers audio in the original text
        order.  For N sentences the total API wait is
        max(t1 … tN)  instead of  t1 + … + tN.
        """
        while True:
            # Block until at least one item arrives
            text = await self._tts_queue.get()
            if text is None:
                self._tts_queue.task_done()
                break

            # Collect batch: current item + anything already queued
            batch = [text]
            while not self._tts_queue.empty():
                try:
                    next_text = self._tts_queue.get_nowait()
                    batch.append(next_text)
                    if next_text is None:       # poison pill
                        break
                except asyncio.QueueEmpty:
                    break

            # Separate real texts from poison pill
            texts: list[str] = []
            has_poison = False
            for item in batch:
                if item is None:
                    has_poison = True
                    break
                texts.append(item)

            # Fire ALL TTS calls in parallel
            tasks = [asyncio.create_task(self.synthesize(t)) for t in texts]

            # Deliver audio strictly in order
            for i, task in enumerate(tasks):
                try:
                    pcm_bytes = await task
                    if pcm_bytes and self._audio_callback:
                        await self._audio_callback(pcm_bytes)
                except Exception:
                    logger.exception("[SarvamTTS] Batch item %d error", i)

            # Mark every dequeued item as done
            for _ in batch:
                self._tts_queue.task_done()

            if has_poison:
                break

    async def synthesize(self, text: str) -> bytes:
        """Convert text to raw PCM audio bytes (24kHz, 16-bit mono).

        Includes timing instrumentation and one automatic retry on timeout.
        """
        if not text or not text.strip():
            return b""

        logger.info("[SarvamTTS] Synthesizing %d chars: %s", len(text), text)

        payload = {
            "text": text,
            "target_language_code": self.target_language,
            "speaker": self.speaker,
            "model": self.model,
            "speech_sample_rate": self.sample_rate,
            "pace": self.pace,
            "temperature": self.temperature,
            "enable_preprocessing": True,
        }
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(2):            # 1 original + 1 retry
            t0 = time.monotonic()
            try:
                response = await self._http_client.post(
                    self.api_url, headers=headers, json=payload,
                )
                elapsed = time.monotonic() - t0

                if response.status_code != 200:
                    logger.error(
                        "[SarvamTTS] API error %d (%.1fs): %s",
                        response.status_code, elapsed, response.text,
                    )
                    response.raise_for_status()

                if elapsed > 3.0:
                    logger.warning(
                        "[SarvamTTS] Slow TTS: %.1fs for %d chars",
                        elapsed, len(text),
                    )
                else:
                    logger.info(
                        "[SarvamTTS] TTS API %.1fs for %d chars",
                        elapsed, len(text),
                    )

                data = response.json()
                audio_b64 = data["audios"][0]
                wav_bytes = base64.b64decode(audio_b64)
                pcm_bytes = self._wav_to_pcm(wav_bytes)
                logger.info("[SarvamTTS] Got %d bytes of PCM audio", len(pcm_bytes))
                return pcm_bytes

            except httpx.TimeoutException:
                elapsed = time.monotonic() - t0
                if attempt == 0:
                    logger.warning(
                        "[SarvamTTS] Timeout after %.1fs, retrying…",
                        elapsed,
                    )
                    continue
                logger.error(
                    "[SarvamTTS] Timeout after %.1fs on retry — giving up",
                    elapsed,
                )
                return b""

        return b""  # unreachable but satisfies type-checker

    async def add_text_delta(self, text_delta: str):
        """Add incremental text from Voice Live response.text.delta.

        Buffers text and queues TTS when a complete sentence is detected.
        """
        self._text_buffer += text_delta

        # Check for sentence boundaries
        await self._try_flush_sentences()

    async def flush_remaining(self):
        """Flush any remaining buffered text through TTS.

        Call this on response.text.done to ensure all text is spoken.
        """
        if self._text_buffer.strip():
            text = self._text_buffer.strip()
            self._text_buffer = ""
            await self._tts_queue.put(text)

        # Wait for the queue to drain (all items processed in order)
        await self._tts_queue.join()

    def clear_buffer(self):
        """Clear the text buffer and cancel pending TTS (e.g., user interrupts)."""
        self._text_buffer = ""
        # Drain the queue without processing
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _try_flush_sentences(self):
        """Check for complete sentences in buffer and queue them for TTS.

        Splits only on sentence-ending punctuation (. ! ? ।) to keep
        sentences whole. Accumulates text until MIN_SENTENCE_LENGTH is
        reached so each TTS call gets enough context for natural prosody.
        """
        parts = SENTENCE_BOUNDARY.split(self._text_buffer)

        if len(parts) > 1:
            # Accumulate complete sentences
            accumulated = ""
            for part in parts[:-1]:
                stripped = part.strip()
                if stripped:
                    accumulated += stripped + " "

            # Keep the last (incomplete) part in the buffer
            last_part = parts[-1]

            if len(accumulated.strip()) >= MIN_SENTENCE_LENGTH:
                # Enough text — queue for sequential TTS
                self._text_buffer = last_part
                await self._tts_queue.put(accumulated.strip())

    async def close(self):
        """Close the persistent HTTP client and stop the worker."""
        try:
            await self._http_client.aclose()
        except Exception:
            logger.debug("[SarvamTTS] Error closing HTTP client", exc_info=True)

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> bytes:
        """Extract raw PCM data from WAV container bytes."""
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                return wav_file.readframes(wav_file.getnframes())
