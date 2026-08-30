"""Minimal Turkish voice chat: Qwen3-Omni via vLLM + Cartesia TTS."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import requests


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000")
MAX_HISTORY_TURNS = max(1, int(os.getenv("MAX_HISTORY_TURNS", "4")))

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3.6")
CARTESIA_VERSION = os.getenv("CARTESIA_VERSION", "2026-08-14")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Sen doğal, sıcak ve yardımcı bir Türkçe sesli asistansın. "
    "Kullanıcının sesli mesajını doğrudan yanıtla ve konuşma bağlamını hatırla. "
    "Her zaman Türkçe konuş. Cevaplarını kısa, gündelik ve sesli okunmaya uygun tut. "
    "Markdown, liste, emoji veya sahne yönergesi kullanma.",
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/qwen3-voice-outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def user_audio_message(audio_path: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": Path(audio_path).resolve().as_uri()},
            },
            {
                "type": "text",
                "text": "Bu sesli mesaja kısa ve doğal bir Türkçe cevap ver.",
            },
        ],
    }


def generate_reply(audio_path: str, history: list[dict]) -> tuple[str, dict]:
    current_message = user_audio_message(audio_path)
    started_at = time.perf_counter()
    response = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                current_message,
            ],
            "modalities": ["text"],
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=(10, 180),
    )
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    if not answer:
        raise RuntimeError("Qwen boş bir yanıt döndürdü.")
    print(f"Qwen süresi: {time.perf_counter() - started_at:.1f}s", flush=True)
    return answer, current_message


def speak_with_cartesia(text: str) -> str:
    started_at = time.perf_counter()
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
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 24_000,
            },
        },
        timeout=(10, 120),
    )
    response.raise_for_status()
    print(f"Cartesia süresi: {time.perf_counter() - started_at:.1f}s", flush=True)

    with tempfile.NamedTemporaryFile(
        prefix="cevap-", suffix=".wav", dir=OUTPUT_DIR, delete=False
    ) as output_file:
        output_file.write(response.content)
        return output_file.name


def talk(
    audio_path: str | None,
    model_history: list[dict] | None,
    display_history: list[dict] | None,
):
    if not audio_path:
        raise gr.Error("Önce bir ses kaydı oluşturun.")
    if not CARTESIA_API_KEY or not CARTESIA_VOICE_ID:
        raise gr.Error(
            "CARTESIA_API_KEY ve CARTESIA_VOICE_ID ortam değişkenlerini ayarlayın."
        )

    history = list(model_history or [])
    try:
        answer, current_message = generate_reply(audio_path, history)
    except requests.RequestException as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise gr.Error(f"Qwen isteği başarısız oldu: {detail}") from exc
    except (KeyError, IndexError, RuntimeError, ValueError) as exc:
        raise gr.Error(f"Qwen yanıtı işlenemedi: {exc}") from exc

    try:
        answer_audio = speak_with_cartesia(answer)
    except requests.RequestException as exc:
        detail = exc.response.text[:300] if exc.response is not None else str(exc)
        raise gr.Error(f"Cartesia isteği başarısız oldu: {detail}") from exc

    history.extend(
        [
            current_message,
            {"role": "assistant", "content": answer},
        ]
    )
    history = history[-(MAX_HISTORY_TURNS * 2) :]

    visible = list(display_history or [])
    visible.extend(
        [
            {"role": "user", "content": "🎙️ Sesli mesaj"},
            {"role": "assistant", "content": answer},
        ]
    )
    return None, visible, answer_audio, history, visible


def reset():
    return None, [], None, [], []


with gr.Blocks(title="Türkçe Qwen3 Sesli Sohbet") as demo:
    gr.Markdown("# Türkçe Sesli Sohbet\nMikrofona konuşun ve **Gönder** düğmesine basın.")
    model_history_state = gr.State([])
    display_history_state = gr.State([])
    chatbot = gr.Chatbot(label="Konuşma", height=520)
    microphone = gr.Audio(label="Mikrofon", sources=["microphone"], type="filepath")
    with gr.Row():
        send_button = gr.Button("Gönder", variant="primary")
        reset_button = gr.Button("Yeni konuşma")
    answer_audio = gr.Audio(label="Yanıt", autoplay=True)

    inputs = [microphone, model_history_state, display_history_state]
    outputs = [
        microphone,
        chatbot,
        answer_audio,
        model_history_state,
        display_history_state,
    ]
    send_button.click(fn=talk, inputs=inputs, outputs=outputs)
    reset_button.click(fn=reset, outputs=outputs)


if __name__ == "__main__":
    print("Arayüz hazır.", flush=True)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=HOST,
        server_port=PORT,
        theme=gr.themes.Soft(),
        allowed_paths=[str(OUTPUT_DIR)],
        show_error=True,
    )
