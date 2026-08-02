FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /alexandria

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY app/requirements.txt /alexandria/app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt && \
    pip install --no-cache-dir qwen-tts==0.1.1 && \
    pip install --no-cache-dir f5-tts ruaccent

# Copy application code
COPY app/ /alexandria/app/
COPY default_prompts.txt review_prompts.txt persona_prompts.txt /alexandria/
COPY builtin_lora/ /alexandria/builtin_lora/

# Only reusable compute assets and disposable jobs live on the server.
RUN mkdir -p /alexandria/lora_models \
    /alexandria/job_workspaces \
    /alexandria/config

# Bind to 0.0.0.0 inside the container
ENV ALEXANDRIA_HOST=0.0.0.0
ENV ALEXANDRIA_CONFIG_PATH=/alexandria/config/config.json
EXPOSE 4200

CMD ["python", "app/app.py"]
