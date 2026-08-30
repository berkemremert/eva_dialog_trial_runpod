FROM qwenllm/qwen3-omni:3-cu124

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade -r requirements.txt
COPY app.py ./

ENV HF_HOME=/workspace/huggingface \
    PORT=7860 \
    HOST=0.0.0.0

EXPOSE 7860
CMD ["python", "app.py"]
