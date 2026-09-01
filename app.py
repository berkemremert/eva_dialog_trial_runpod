"""Realtime Turkish voice chat: FastRTC VAD + Qwen3-Omni + Cartesia TTS."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
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
    get_cloudflare_turn_credentials,
    get_cloudflare_turn_credentials_async,
    get_current_context,
)


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000")
MAX_HISTORY_TURNS = max(1, int(os.getenv("MAX_HISTORY_TURNS", "4")))
MAX_RESPONSE_TOKENS = max(32, int(os.getenv("MAX_RESPONSE_TOKENS", "96")))
TURN_PROVIDER = os.getenv("TURN_PROVIDER", "openrelay").strip().lower()

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3.6")
CARTESIA_VERSION = os.getenv("CARTESIA_VERSION", "2026-08-14")

VAD_CHUNK_SECONDS = float(os.getenv("VAD_CHUNK_SECONDS", "0.4"))
VAD_START_SECONDS = float(os.getenv("VAD_START_SECONDS", "0.15"))
VAD_STOP_SECONDS = float(os.getenv("VAD_STOP_SECONDS", "0.08"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "350"))

BASE_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Her zaman Türkçe konuş. Telefonda doğal, sıcak ve profesyonel davran. "
    "Her turda tek veya en fazla iki kısa cümle söyle ve yalnızca bir soru sor. "
    "Markdown, liste, emoji veya sahne yönergesi kullanma. "
    "Şifre, PIN, kartın tamamı veya tek kullanımlık kod gibi hassas bilgi isteme. "
    "Yanıtını yalnızca geçerli JSON olarak şu biçimde döndür: "
    '{"transcript":"kullanıcının söylediği","answer":"senin cevabın"}.',
)

PERSONAS = {
    "Banka satış temsilcisi": {
        "opening": (
            "Merhaba, ben Nova Bank müşteri ekibinden Ece. "
            "Size uygun kart avantajlarını kısaca anlatmak için aradım, müsait misiniz?"
        ),
        "instructions": (
            "Kurgusal Nova Bank adına kredi kartı ve bankacılık ürünleri sunan bir satış "
            "temsilcisisin. İhtiyacı keşfet, faydayı kısa anlat ve nazikçe sonraki adıma "
            "ilerle. Kesin onay veya getiri sözü verme ve hassas finansal bilgi isteme."
        ),
    },
    "Otel resepsiyonisti": {
        "opening": (
            "Merhaba, Mavi Kıyı Oteli resepsiyonundan Deniz ben. "
            "Yaklaşan konaklamanızla ilgili birkaç detayı teyit etmek için arıyorum, müsait misiniz?"
        ),
        "instructions": (
            "Kurgusal Mavi Kıyı Oteli'nin resepsiyonistisin. Rezervasyon tarihini, kişi "
            "sayısını, oda tercihini ve özel talepleri doğal bir telefon görüşmesiyle teyit et."
        ),
    },
    "Telekom satış temsilcisi": {
        "opening": (
            "Merhaba, Atlas İletişim'den Arda ben. "
            "Mevcut kullanımınıza uygun yeni tarifemizi paylaşmak için aradım, müsait misiniz?"
        ),
        "instructions": (
            "Kurgusal Atlas İletişim adına tarife sunan bir satış temsilcisisin. Kullanım "
            "ihtiyacını sor, uygun paketi kısa ve açık biçimde anlat, baskıcı davranma."
        ),
    },
    "Genel müşteri temsilcisi": {
        "opening": (
            "Merhaba, müşteri deneyimi ekibinden Selin ben. "
            "Kısa bir hizmet görüşmesi için aradım, şu anda müsait misiniz?"
        ),
        "instructions": (
            "Bir şirketin müşteri deneyimi temsilcisisin. Kullanıcının ihtiyacını keşfet, "
            "sorununu netleştir ve kısa, çözüm odaklı bir görüşme yürüt."
        ),
    },
}
DEFAULT_PERSONA = "Banka satış temsilcisi"

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "/tmp/qwen3-realtime-audio"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Turn:
    transcript: str
    answer: str


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    persona: str = DEFAULT_PERSONA
    opening: str = ""
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


def persona_config(persona: str | None) -> tuple[str, dict[str, str]]:
    selected = persona if persona in PERSONAS else DEFAULT_PERSONA
    return selected, PERSONAS[selected]


def persona_system_prompt(persona: str) -> str:
    selected, config = persona_config(persona)
    return (
        f"{BASE_SYSTEM_PROMPT} Senaryo: {selected}. {config['instructions']} "
        f"Görüşmeyi daha önce şu cümlelerle açtın: {config['opening']} "
        "Kullanıcının yeni sesini doğru yazıya dök ve bu telefon görüşmesini kaldığı "
        "yerden sürdür. JSON dışına hiçbir şey yazma."
    )


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
                "text": (
                    "Bu ses kaydındaki Türkçe konuşmayı transcript alanına yaz ve telefon "
                    "görüşmesine uygun kısa cevabı answer alanına yaz. Yalnızca JSON döndür."
                ),
            },
        ],
    }


def qwen_messages(
    previous_turns: list[Turn], audio_path: Path, persona: str
) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": persona_system_prompt(persona)}
    ]
    for turn in previous_turns:
        messages.extend(
            [
                {"role": "user", "content": turn.transcript},
                {"role": "assistant", "content": turn.answer},
            ]
        )
    messages.append(audio_message(audio_path))
    return messages


def parse_qwen_result(raw_answer: str) -> tuple[str, str]:
    candidate = raw_answer.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
    json_start = candidate.find("{")
    json_end = candidate.rfind("}")
    if json_start >= 0 and json_end > json_start:
        candidate = candidate[json_start : json_end + 1]
    try:
        payload = json.loads(candidate)
        transcript = str(payload.get("transcript", "")).strip()
        answer = str(payload.get("answer", "")).strip()
    except (TypeError, ValueError, json.JSONDecodeError):
        transcript = "Sesli mesaj"
        answer = raw_answer.strip()
    if not answer:
        raise RuntimeError("Qwen boş bir yanıt döndürdü.")
    return transcript or "Sesli mesaj", answer


def generate_reply(
    audio_path: Path, previous_turns: list[Turn], persona: str
) -> tuple[str, str]:
    started_at = time.perf_counter()
    print("Qwen isteği başladı.", flush=True)
    response = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": qwen_messages(previous_turns, audio_path, persona),
            "modalities": ["text"],
            "max_tokens": MAX_RESPONSE_TOKENS,
            "temperature": 0,
        },
        timeout=(10, 180),
    )
    response.raise_for_status()
    transcript, answer = parse_qwen_result(
        response.json()["choices"][0]["message"]["content"]
    )
    elapsed = time.perf_counter() - started_at
    print(
        f"Qwen süresi: {elapsed:.1f}s | transcript={transcript!r} | cevap={answer!r}",
        flush=True,
    )
    return transcript, answer


TURKISH_NUMBER_PATTERN = re.compile(
    r"(?<![\w])[-+]?\d{1,3}(?:\.\d{3})+(?:,\d+)?(?![\w])"
    r"|(?<![\w])[-+]?\d+(?:[.,]\d+)?(?![\w])"
)
TURKISH_ONES = (
    "",
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    "altı",
    "yedi",
    "sekiz",
    "dokuz",
)
TURKISH_TENS = (
    "",
    "on",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
)
TURKISH_SCALES = ("", "bin", "milyon", "milyar", "trilyon", "katrilyon")
TURKISH_MONTHS = (
    "",
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
)


def turkish_integer_to_words(number: int) -> str:
    if number == 0:
        return "sıfır"
    if number < 0:
        return f"eksi {turkish_integer_to_words(-number)}"

    groups: list[str] = []
    scale_index = 0
    while number:
        group = number % 1000
        if group:
            words: list[str] = []
            hundreds, remainder = divmod(group, 100)
            tens, ones = divmod(remainder, 10)
            if hundreds:
                if hundreds > 1:
                    words.append(TURKISH_ONES[hundreds])
                words.append("yüz")
            if tens:
                words.append(TURKISH_TENS[tens])
            if ones:
                words.append(TURKISH_ONES[ones])

            scale = TURKISH_SCALES[scale_index]
            if scale:
                if not (scale == "bin" and group == 1):
                    words.append(scale)
                else:
                    words = ["bin"]
            groups.append(" ".join(words))
        number //= 1000
        scale_index += 1
        if scale_index >= len(TURKISH_SCALES) and number:
            # Phone/card-like extremely long values sound clearer digit by digit.
            prefix = " ".join(TURKISH_ONES[int(digit)] or "sıfır" for digit in str(number))
            groups.append(prefix)
            break
    return " ".join(reversed(groups))


def turkish_number_to_words(raw: str) -> str:
    sign = ""
    if raw.startswith(("-", "+")):
        sign = "eksi " if raw[0] == "-" else "artı "
        raw = raw[1:]

    if raw.isdigit() and len(raw) >= 7:
        return sign + " ".join(
            TURKISH_ONES[int(digit)] or "sıfır" for digit in raw
        )

    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", raw):
        integer_part, _, decimal_part = raw.partition(",")
        integer_part = integer_part.replace(".", "")
    else:
        separator = "," if "," in raw else "." if "." in raw else ""
        if separator:
            integer_part, decimal_part = raw.split(separator, 1)
        else:
            integer_part, decimal_part = raw, ""

    spoken = sign + turkish_integer_to_words(int(integer_part or "0"))
    if decimal_part:
        if decimal_part.startswith("0"):
            decimal_words = " ".join(
                turkish_integer_to_words(int(digit)) for digit in decimal_part
            )
        else:
            decimal_words = turkish_integer_to_words(int(decimal_part))
        spoken += f" virgül {decimal_words}"
    return spoken


def normalize_turkish_for_tts(text: str) -> str:
    """Make cardinal numbers explicit so TTS says 100 as 'yüz', not digit by digit."""
    def replace_date(match: re.Match[str]) -> str:
        day, month, year = (int(part) for part in match.groups())
        if not 1 <= month <= 12:
            return match.group(0)
        return (
            f"{turkish_integer_to_words(day)} {TURKISH_MONTHS[month]} "
            f"{turkish_integer_to_words(year)}"
        )

    normalized = re.sub(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", replace_date, text)
    normalized = re.sub(r"%\s*", "yüzde ", normalized)
    normalized = re.sub(r"(?i)\bTL\b", "Türk lirası", normalized)
    normalized = normalized.replace("₺", " Türk lirası")
    normalized = TURKISH_NUMBER_PATTERN.sub(
        lambda match: turkish_number_to_words(match.group(0)), normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def stream_cartesia(text: str) -> Iterator[tuple[int, np.ndarray]]:
    """Stream raw 24 kHz PCM so playback can begin early and be interrupted."""
    started_at = time.perf_counter()
    spoken_text = normalize_turkish_for_tts(text)
    if spoken_text != text:
        print(f"TTS sayı normalizasyonu: {text!r} -> {spoken_text!r}", flush=True)
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
            "transcript": spoken_text,
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


def display_messages(conversation: Conversation) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if conversation.opening:
        messages.append({"role": "assistant", "content": conversation.opening})
    for turn in conversation.turns:
        messages.extend(
            [
                {"role": "user", "content": turn.transcript},
                {"role": "assistant", "content": turn.answer},
            ]
        )
    return messages


def prepare_conversation(
    conversation: Conversation, persona: str
) -> tuple[str, dict[str, str]]:
    selected, config = persona_config(persona)
    if conversation.persona != selected:
        conversation.turns.clear()
        conversation.persona = selected
        conversation.opening = config["opening"]
    elif not conversation.opening:
        conversation.opening = config["opening"]
    return selected, config


def startup(persona: str = DEFAULT_PERSONA):
    if not CARTESIA_API_KEY or not CARTESIA_VOICE_ID:
        raise RuntimeError(
            "CARTESIA_API_KEY ve CARTESIA_VOICE_ID ortam değişkenlerini ayarlayın."
        )
    selected, config = persona_config(persona)
    opening = config["opening"]
    current_display = [{"role": "assistant", "content": opening}]
    print(f"Görüşme başladı: {selected}", flush=True)
    yield AdditionalOutputs(current_display)
    yield from stream_cartesia(opening)


def respond(audio: tuple[int, np.ndarray], persona: str = DEFAULT_PERSONA):
    if not CARTESIA_API_KEY or not CARTESIA_VOICE_ID:
        raise RuntimeError(
            "CARTESIA_API_KEY ve CARTESIA_VOICE_ID ortam değişkenlerini ayarlayın."
        )

    context = get_current_context()
    conversation = conversation_for(context.webrtc_id)
    sample_rate, samples = audio
    audio_seconds = np.asarray(samples).size / max(sample_rate, 1)
    print(f"Konuşma algılandı: {audio_seconds:.1f}s", flush=True)
    audio_path = save_wav(audio)

    # One Qwen request at a time per browser conversation keeps history ordered.
    with conversation.lock:
        try:
            selected, _config = prepare_conversation(conversation, persona)
            transcript, answer = generate_reply(
                audio_path, list(conversation.turns), selected
            )
        finally:
            audio_path.unlink(missing_ok=True)

        conversation.turns.append(Turn(transcript=transcript, answer=answer))
        while len(conversation.turns) > MAX_HISTORY_TURNS:
            conversation.turns.pop(0)
        current_display = display_messages(conversation)

    # Show the text immediately, then stream audio in small interruptible chunks.
    yield AdditionalOutputs(current_display)
    yield from stream_cartesia(answer)


def openrelay_rtc_configuration(ttl: int = 3600) -> dict:
    """Create temporary credentials for Metered's public static-auth test relay."""
    username = f"{int(time.time()) + ttl}:qwen3-runpod"
    credential = base64.b64encode(
        hmac.new(
            b"openrelayprojectsecret",
            username.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode("ascii")
    return {
        "iceServers": [
            {"urls": ["stun:stun.cloudflare.com:3478"]},
            {
                "urls": [
                    "turn:staticauth.openrelay.metered.ca:80?transport=udp",
                    "turn:staticauth.openrelay.metered.ca:80?transport=tcp",
                    "turns:staticauth.openrelay.metered.ca:443?transport=tcp",
                ],
                "username": username,
                "credential": credential,
            },
        ]
    }


def cloudflare_turn_kwargs() -> dict:
    key_id = os.getenv("CLOUDFLARE_TURN_KEY_ID", "").strip()
    api_token = os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN", "").strip()
    if not key_id or not api_token:
        raise RuntimeError(
            "TURN_PROVIDER=cloudflare için CLOUDFLARE_TURN_KEY_ID ve "
            "CLOUDFLARE_TURN_KEY_API_TOKEN gerekli."
        )
    # hf_token is deliberately blank: turn.fastrtc.org currently has no DNS.
    return {
        "turn_key_id": key_id,
        "turn_key_api_token": api_token,
        "hf_token": "",
    }


async def logged_turn_credentials():
    """Return client TURN configuration and log complete connection failures."""
    try:
        if TURN_PROVIDER == "openrelay":
            print("TURN sağlayıcısı: Metered Open Relay (test)", flush=True)
            credentials = openrelay_rtc_configuration()
        elif TURN_PROVIDER == "cloudflare":
            print("TURN sağlayıcısı: doğrudan Cloudflare", flush=True)
            credentials = await get_cloudflare_turn_credentials_async(
                **cloudflare_turn_kwargs(), ttl=3600
            )
        else:
            raise RuntimeError(
                f"Bilinmeyen TURN_PROVIDER={TURN_PROVIDER!r}; "
                "openrelay veya cloudflare kullanın."
            )
    except Exception:
        print("TURN kimlik bilgileri alınamadı. Tam hata:", flush=True)
        traceback.print_exc()
        raise
    print("TURN kimlik bilgileri hazır.", flush=True)
    return credentials


def server_turn_configuration() -> dict:
    if TURN_PROVIDER == "openrelay":
        return openrelay_rtc_configuration(ttl=86_400)
    if TURN_PROVIDER == "cloudflare":
        return get_cloudflare_turn_credentials(
            **cloudflare_turn_kwargs(), ttl=86_400
        )
    raise RuntimeError(
        f"Bilinmeyen TURN_PROVIDER={TURN_PROVIDER!r}; openrelay veya cloudflare kullanın."
    )


rtc_configuration = logged_turn_credentials
server_rtc_configuration = server_turn_configuration()

chatbot = gr.Chatbot(label="Konuşma", type="messages", height=420)
persona_selector = gr.Dropdown(
    choices=list(PERSONAS),
    value=DEFAULT_PERSONA,
    label="Arama senaryosu",
    info="Görüşmeyi başlatmadan önce rolü seçin.",
)
voice_stream = Stream(
    handler=ReplyOnPause(
        respond,
        startup_fn=startup,
        can_interrupt=True,
        input_sample_rate=16_000,
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
    server_rtc_configuration=server_rtc_configuration,
    concurrency_limit=1,
    additional_inputs=[persona_selector],
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
    print(f"TURN yapılandırıldı: {TURN_PROVIDER}", flush=True)
    print("Realtime arayüz hazır.", flush=True)
    voice_stream.ui.launch(
        server_name=HOST,
        server_port=PORT,
        show_error=True,
    )
