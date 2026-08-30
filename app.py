"""Minimal Turkish voice chat: Qwen3-Omni audio understanding + Cartesia TTS."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import requests
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
ATTENTION = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")
MAX_HISTORY_TURNS = max(1, int(os.getenv("MAX_HISTORY_TURNS", "6")))

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


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı. GPU etkin bir RunPod Pod kullanın.")

    options = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": ATTENTION,
    }
    try:
        loaded_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_ID, **options
        )
    except (ImportError, RuntimeError) as exc:
        if ATTENTION != "flash_attention_2" or "flash" not in str(exc).lower():
            raise
        print(f"FlashAttention 2 kullanılamıyor ({exc}); SDPA kullanılacak.")
        options["attn_implementation"] = "sdpa"
        loaded_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_ID, **options
        )

    # Cartesia speaks the response, so Qwen's talker is unnecessary.
    loaded_model.disable_talker()
    loaded_model.eval()
    loaded_processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
    print("Model hazır.")
    return loaded_model, loaded_processor


def generate_reply(audio_path: str, history: list[dict]) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        *history,
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=rendered,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.inference_mode():
        text_ids, _ = model.generate(
            **inputs,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=256,
            thinker_do_sample=True,
            thinker_temperature=0.6,
            thinker_top_p=0.9,
            thinker_top_k=20,
            use_audio_in_video=False,
        )

    generated = text_ids.sequences[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def speak_with_cartesia(text: str) -> str:
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
        answer = generate_reply(audio_path, history)
        answer_audio = speak_with_cartesia(answer)
    except requests.RequestException as exc:
        detail = exc.response.text[:300] if exc.response is not None else str(exc)
        raise gr.Error(f"Cartesia isteği başarısız oldu: {detail}") from exc

    # Keep the original audio in Qwen's recent context and the assistant's text response.
    history.extend(
        [
            {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
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


model, processor = load_model()

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
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=HOST,
        server_port=PORT,
        theme=gr.themes.Soft(),
        allowed_paths=[str(OUTPUT_DIR)],
        show_error=True,
    )
