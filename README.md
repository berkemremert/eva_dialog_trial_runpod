# Turkish Qwen3 voice chat on RunPod

A minimal voice-conversation prototype:

```text
Microphone -> Qwen3-Omni audio understanding -> Cartesia Turkish TTS -> Speaker
```

Qwen3-Omni consumes the Turkish audio directly through a two-GPU vLLM tensor-parallel server and
writes the response. Cartesia is used only to speak that response in Turkish. vLLM serves only
Qwen's thinker because the built-in talker is unnecessary here.

This version is intentionally voice-turn based: record, press Send, and hear the answer. It is a
small first step for testing dialogue and voice quality before adding continuous audio streaming,
automatic turn detection, or interruption handling.

## RunPod setup

Create a GPU Pod with:

- Container image: RunPod PyTorch 2.4 / CUDA 12.4
- Two A40 48 GB GPUs (96 GB total VRAM)
- At least 120 GB of persistent storage
- HTTP port `7860` exposed

Create a Cartesia API key and choose or localize a voice for Turkish. Then open the RunPod terminal:

```bash
cd /workspace
git clone YOUR_REPOSITORY_URL qwen3-voice
cd qwen3-voice

export CARTESIA_API_KEY="sk_car_..."
export CARTESIA_VOICE_ID="your-turkish-voice-id"
bash start.sh
```

The first start installs the pinned vLLM runtime and downloads the model. When `Qwen hazır` and `Arayüz hazır` appear,
open port 7860 from RunPod's
Connect page. The model cache defaults to `/workspace/huggingface`, so it survives Pod restarts when
that directory is backed by persistent storage.

Never commit your Cartesia key to the repository.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CARTESIA_API_KEY` | required | Cartesia API key |
| `CARTESIA_VOICE_ID` | required | Turkish-capable or Turkish-localized Cartesia voice |
| `CARTESIA_MODEL` | `sonic-3.6` | Cartesia TTS model |
| `CARTESIA_VERSION` | `2026-08-14` | Cartesia API version |
| `MODEL_ID` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Qwen model ID or local path |
| `MAX_HISTORY_TURNS` | `6` | Recent voice turns retained for Qwen |
| `SYSTEM_PROMPT` | built-in Turkish prompt | Assistant persona and behavior |
| `HF_HOME` | `/workspace/huggingface` | Model cache when using `start.sh` |
| `PORT` | `7860` | Web interface port |
| `VLLM_PORT` | `8000` | Internal-only Qwen API port |

## What was intentionally removed

- Image and video inputs
- Text input
- Cartesia speech-to-text
- Qwen's built-in speech output
- Voice and sampling controls in the UI
- Long-lived or unlimited conversation history

The remaining interface has one microphone, one Send button, the conversation, and the latest
spoken response.
