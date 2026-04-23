FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-baixa o modelo buffalo_l para dentro da imagem
# Evita download na primeira requisição em produção
RUN python -c "
import insightface
m = insightface.app.FaceAnalysis(name='buffalo_l')
m.prepare(ctx_id=-1)
print('Model downloaded successfully')
"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]