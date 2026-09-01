FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=8s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/health', timeout=6)" || exit 1

CMD ["python3", "-u", "deploy_api.py"]
