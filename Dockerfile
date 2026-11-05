# Lightweight build for small free hosts (e.g. Render free, 512 MB RAM).
# No PyTorch -> Words mode is disabled, Letters mode (fingerspelling) still works.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-lite.txt .
RUN pip install --no-cache-dir -r requirements-lite.txt

COPY . .

# Render sets $PORT (usually 10000)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
