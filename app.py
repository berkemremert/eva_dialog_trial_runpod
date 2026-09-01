"""Realtime Turkish voice chat: FastRTC VAD + Qwen3-Omni + Cartesia TTS."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import gradio as gr
import numpy as np
import requests
from fastrtc import (
    AdditionalOutputs,
    AlgoOptions,
    ReplyOnPause,
    SileroVadOptions,
    Stream,
    get_cloudflare_turn_credentials_async,
    get_current_context,
)


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000")
MAX_HISTORY_TURNS = max(1, int(os.getenv("MAX_HISTORY_TURNS", "4")))

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3.6")
CARTESIA_VERSION = os.getenv("CARTESIA_VERSION", "2026-08-14")

VAD_CHUNK_SECONDS = float(os.getenv("VAD_CHUNK_SECONDS", "0.6"))
VAD_START_SECONDS = float(os.getenv("VAD_START_SECONDS", "0.15"))
VAD_STOP_SECONDS = float(os.getenv("VAD_STOP_SECONDS", "0.10"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "500"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Sen doğal, sıcak ve yardımcı bir Türkçe sesli asistansın. "
    "Kullanıcının sesli mesajını doğrudan yanıtla ve konuşma bağlamını hatırla. "
    "Her zaman Türkçe konuş. Cevaplarını kısa, gündelik ve sesli okunmaya uygun tut. "
    "Markdown, liste, emoji veya sahne yönergesi kullanma.",
)

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "/tmp/qwen3-realtime-audio"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Turn:
    audio_path: Path
    answer: str


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


CONVERSATIONS: dict[str, Conversation] = {}
CONVERSATIONS_LOCK = threading.Lock()


def conversation_for(connection_id: str) -> Conversation:
    with CONVERSATIONS_LOCK:
        return CONVERSATIONS.setdefault(connection_id, Conversation())


def save_wav(audio: tuple[int, np.ndarray]) -> Path:
    sample_rate, samples = audio
    mono = np.asarray(samples).reshape(-1)
    if np.issubdtype(mono.dtype, np.floating):
        mono = np.clip(mono, -1.0, 1.0)
        mono = (mono * 32767).astype(np.int16)
    else:
        mono = mono.astype(np.int16, copy=False)

    with tempfile.NamedTemporaryFile(
        prefix="utterance-", suffix=".wav", dir=AUDIO_DIR, delete=False
    ) as audio_file:
        output_path = Path(audio_file.name)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(mono.tobytes())
    return output_path


def audio_message(audio_path: Path) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_path.resolve().as_uri()},
            },
            {
                "type": "text",
                "text": "Bu sesli mesaja kısa ve doğal bir Türkçe cevap ver.",
            },
        ],
    }


def qwen_messages(previous_turns: list[Turn], audio_path: Path) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in previous_turns:
        messages.extend(
            [
                audio_message(turn.audio_path),
                {"role": "assistant", "content": turn.answer},
            ]
        )
    messages.append(audio_message(audio_path))
    return messages


def generate_reply(audio_path: Path, previous_turns: list[Turn]) -> str:
    started_at = time.perf_counter()
    print("Qwen isteği başladı.", flush=True)
    response = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": qwen_messages(previous_turns, audio_path),
            "modalities": ["text"],
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=(10, 180),
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()
    if not answer:
        raise RuntimeError("Qwen boş bir yanıt döndürdü.")
    print(f"Qwen süresi: {time.perf_counter() - started_at:.1f}s", flush=True)
    return answer


def stream_cartesia(text: str) -> Iterator[tuple[int, np.ndarray]]:
    """Stream raw 24 kHz PCM so playback can begin early and be interrupted."""
    started_at = time.perf_counter()
    print("Cartesia akışı başladı.", flush=True)
    response = requests.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Cartesia-Version": CARTESIA_VERSION,
            "Content-Type": "application/json",
        },
        json={
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": CARTESIA_VOICE_ID,
            "language": "tr",
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 24_000,
            },
        },
        stream=True,
        timeout=(10, 120),
    )
    try:
        response.raise_for_status()
        pending = b""
        first_chunk = True
        for chunk in response.iter_content(chunk_size=1920):
            if not chunk:
                continue
            if first_chunk:
                print(
                    f"Cartesia ilk ses: {time.perf_counter() - started_at:.1f}s",
                    flush=True,
                )
                first_chunk = False
            pending += chunk
            usable_bytes = len(pending) - (len(pending) % 2)
            if usable_bytes:
                pcm = np.frombuffer(pending[:usable_bytes], dtype="<i2").copy()
                pending = pending[usable_bytes:]
                yield 24_000, pcm.reshape(1, -1)
        print(f"Cartesia süresi: {time.perf_counter() - started_at:.1f}s", flush=True)
    finally:
        # ReplyOnPause closes this generator when the user interrupts.
        response.close()


def display_messages(turns: list[Turn]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.extend(
            [
                {"role": "user", "content": "🎙️ Sesli mesaj"},
                {"role": "assistant", "content": turn.answer},
            ]
        )
    return messages


def respond(audio: tuple[int, np.ndarray]):
    if not CARTESIA_API_KEY or not CARTESIA_VOICE_ID:
        raise RuntimeError(
            "CARTESIA_API_KEY ve CARTESIA_VOICE_ID ortam değişkenlerini ayarlayın."
        )

    context = get_current_context()
    conversation = conversation_for(context.webrtc_id)
    audio_path = save_wav(audio)

    # One Qwen request at a time per browser conversation keeps history ordered.
    with conversation.lock:
        try:
            answer = generate_reply(audio_path, list(conversation.turns))
        except Exception:
            audio_path.unlink(missing_ok=True)
            raise

        conversation.turns.append(Turn(audio_path=audio_path, answer=answer))
        while len(conversation.turns) > MAX_HISTORY_TURNS:
            expired = conversation.turns.pop(0)
            expired.audio_path.unlink(missing_ok=True)
        current_display = display_messages(conversation.turns)

    # Show the text immediately, then stream audio in small interruptible chunks.
    yield AdditionalOutputs(current_display)
    yield from stream_cartesia(answer)


has_turn_credentials = bool(
    os.getenv("HF_TOKEN")
    or (
        os.getenv("CLOUDFLARE_TURN_KEY_ID")
        and os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN")
    )
)


async def logged_turn_credentials():
    """Expose the complete TURN failure in the Pod log instead of only the UI toast."""
    print("TURN kimlik bilgileri isteniyor: https://turn.fastrtc.org/credentials", flush=True)
    try:
        credentials = await get_cloudflare_turn_credentials_async()
    except Exception:
        print("TURN kimlik bilgileri alınamadı. Tam hata:", flush=True)
        traceback.print_exc()
        raise
    print("TURN kimlik bilgileri hazır.", flush=True)
    return credentials


rtc_configuration = (
    logged_turn_credentials if has_turn_credentials else None
)

chatbot = gr.Chatbot(label="Konuşma", type="messages", height=420)
voice_stream = Stream(
    handler=ReplyOnPause(
        respond,
        can_interrupt=True,
        output_sample_rate=24_000,
        algo_options=AlgoOptions(
            audio_chunk_duration=VAD_CHUNK_SECONDS,
            started_talking_threshold=VAD_START_SECONDS,
            speech_threshold=VAD_STOP_SECONDS,
        ),
        model_options=SileroVadOptions(
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        ),
    ),
    modality="audio",
    mode="send-receive",
    rtc_configuration=rtc_configuration,
    concurrency_limit=1,
    additional_outputs=[chatbot],
    additional_outputs_handler=lambda _old, new: new,
    ui_args={
        "title": "Türkçe Qwen3 Realtime Sohbet",
        "subtitle": (
            "Konuşmaya başlayın; durduğunuzda otomatik yanıt verir. "
            "Yanıtı kesmek için tekrar konuşun."
        ),
    },
)


if __name__ == "__main__":
    if not has_turn_credentials:
        print(
            "TURN yapılandırılmadı. RunPod'da bağlantı kurulmazsa HF_TOKEN ayarlayın.",
            flush=True,
        )
    print("Realtime arayüz hazır.", flush=True)
    voice_stream.ui.launch(
        server_name=HOST,
        server_port=PORT,
        show_error=True,
    )
