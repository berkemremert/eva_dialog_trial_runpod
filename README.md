# Turkish Qwen3 voice chat on RunPod

A realtime voice-conversation prototype:

```text
Microphone -> Qwen3-Omni audio understanding -> Cartesia Turkish TTS -> Speaker
```

Qwen3-Omni consumes Turkish audio directly through a vLLM server and writes
the response. Cartesia is used only to stream that response as Turkish speech. FastRTC and Silero
VAD keep the microphone open, detect when the user has finished speaking, and stop the current
response when the user starts talking again.

## RunPod setup

Create a GPU Pod with:

- Container image: RunPod PyTorch 2.4 / CUDA 12.4
- One A100 SXM 80 GB GPU
- At least 120 GB of persistent storage
- HTTP port `7860` exposed

Create a Cartesia API key and choose or localize a voice for Turkish. Then open the RunPod terminal:

```bash
cd /workspace
git clone YOUR_REPOSITORY_URL qwen3-voice
cd qwen3-voice

export CARTESIA_API_KEY="sk_car_..."
export CARTESIA_VOICE_ID="your-turkish-voice-id"
# Optional: improves Hugging Face model-download rate limits only.
export HF_TOKEN="hf_..."
bash start.sh
```

The first start installs the pinned vLLM runtime in an isolated environment under
`/workspace/.venvs` and downloads the model. This avoids dependency conflicts with the Gradio UI.
When `Qwen hazır` and `Realtime arayüz hazır` appear,
open port 7860 from RunPod's
Connect page. The model cache defaults to `/workspace/huggingface`, so it survives Pod restarts when
that directory is backed by persistent storage. Once vLLM is healthy, later `bash start.sh` runs
reuse it; restarting only the interface no longer reloads the model weights.

Never commit your Cartesia key to the repository.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CARTESIA_API_KEY` | required | Cartesia API key |
| `CARTESIA_VOICE_ID` | required | Turkish-capable or Turkish-localized Cartesia voice |
| `CARTESIA_MODEL` | `sonic-3.6` | Cartesia TTS model |
| `CARTESIA_VERSION` | `2026-08-14` | Cartesia API version |
| `MODEL_ID` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Qwen model ID or local path |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs used by vLLM |
| `MAX_MODEL_LEN` | `4096` | Maximum conversation context length |
| `MAX_HISTORY_TURNS` | `4` | Recent voice turns retained for Qwen |
| `SYSTEM_PROMPT` | built-in Turkish prompt | Assistant persona and behavior |
| `HF_HOME` | `/workspace/huggingface` | Model cache when using `start.sh` |
| `PORT` | `7860` | Web interface port |
| `VLLM_PORT` | `8000` | Internal-only Qwen API port |
| `HF_TOKEN` | optional | Improves Hugging Face model-download rate limits |
| `TURN_PROVIDER` | `openrelay` | `openrelay` for testing or `cloudflare` for production |
| `CLOUDFLARE_TURN_KEY_ID` | required for `cloudflare` | Cloudflare TURN key ID |
| `CLOUDFLARE_TURN_KEY_API_TOKEN` | required for `cloudflare` | Cloudflare TURN key secret |
| `VAD_CHUNK_SECONDS` | `0.6` | Amount of recent audio considered for turn detection |
| `VAD_START_SECONDS` | `0.15` | Speech required to treat input as a real interruption |
| `VAD_STOP_SECONDS` | `0.10` | Maximum speech in a chunk before it counts as a pause |
| `VAD_THRESHOLD` | `0.5` | Silero speech confidence threshold |
| `VAD_MIN_SILENCE_MS` | `500` | Silence used to mark the end of a turn |

## What was intentionally removed

- Image and video inputs
- Manual recording and Send button
- Cartesia speech-to-text
- Qwen's built-in speech output
- Voice and sampling controls in the UI
- Long-lived or unlimited conversation history

The remaining interface is a continuous WebRTC call with automatic turn detection, interruptible
streaming speech, and a short conversation transcript.

## Realtime behavior

1. Start the WebRTC session and speak normally.
2. Roughly half a second of silence ends the turn automatically.
3. Qwen receives the complete utterance as audio and produces a short Turkish reply.
4. Cartesia audio begins playing as soon as raw PCM chunks arrive.
5. Speaking during playback cancels the current audio generator and starts a new turn.

Headphones give the most reliable interruption behavior because loudspeaker echo can otherwise be
mistaken for the user speaking. The default Metered Open Relay configuration is intended only for
prototype testing. For production, create a Cloudflare TURN key, set both Cloudflare variables,
and set `TURN_PROVIDER=cloudflare`. The obsolete Hugging Face/FastRTC relay is intentionally not
used because its `turn.fastrtc.org` hostname currently fails DNS resolution.
