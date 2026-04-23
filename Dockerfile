FROM python:3.11-slim

# Wheels primeiro = sem compilar insightface (bem mais rápido em linux/amd64)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --prefer-binary: usa .whl do PyPI; evita `Building wheel for insightface` (vários minutos)
# build-essential só se algum pacote não tiver wheel na tua plataforma
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]