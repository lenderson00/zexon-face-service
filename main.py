import os

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import insightface
import numpy as np
import cv2

# Pacote padrão: antelopev2 (ResNet100@Glint360K) — melhor opção “plug and play” no zoo público
# (incl. casos difíceis: óculos, oclusão parcial, máscara; não é perfeita sem multi-cadastro).
# Embeddings mudam de modelo: trocar o pacote exige recadastrar.
# Alternativa mais leve / download às vezes mais simples: INSIGHTFACE_MODEL=buffalo_l
_MODEL_NAME = os.environ.get("INSIGHTFACE_MODEL", "antelopev2")
# -1 = CPU; 0,1... = GPU CUDA
_CTX_ID = int(os.environ.get("INSIGHTFACE_CTX", "-1"))
# (640,640) padrão; (1280,1280) se o rosto for muito pequeno no enquadramento
_DET = tuple(int(x) for x in os.environ.get("INSIGHTFACE_DET_SIZE", "640,640").split(","))
if len(_DET) != 2:
    _DET = (640, 640)
# 0.45–0.5 típico; menor = mais detecções (e mais falso positivo)
_DET_THRESH = float(os.environ.get("INSIGHTFACE_DET_THRESH", "0.5"))


def _onnxruntime_providers() -> list[str]:
    custom = os.environ.get("ONNXRUNTIME_PROVIDERS")
    if custom:
        return [p.strip() for p in custom.split(",") if p.strip()]
    if _CTX_ID < 0:
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


ORT_PROVIDERS = _onnxruntime_providers()

app = FastAPI(title="Zexon Face Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Só detecção + reconhecimento (sem gender/age) — menos memória, mesmo embedding
model = insightface.app.FaceAnalysis(
    name=_MODEL_NAME,
    allowed_modules=["detection", "recognition"],
    providers=ORT_PROVIDERS,
)
model.prepare(
    ctx_id=_CTX_ID,
    det_size=_DET,
    det_thresh=_DET_THRESH,
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _MODEL_NAME,
        "det_size": list(_DET),
        "det_thresh": _DET_THRESH,
        "ctx_id": _CTX_ID,
        "onnx_providers": ORT_PROVIDERS,
        "dica_reconhecimento": (
            "O pacote aberto mais forte para uso geral é antelopev2. "
            "Com máscara, guarde dois embeddings por pessoa (com e sem) e use o melhor cosseno; "
            "nenhum modelo open gratuito evita 100% falhas só com uma foto de cadastro."
        ),
    }

@app.post("/embed")
async def embed(file: UploadFile):
    # Lê os bytes da imagem
    data = await file.read()
    
    # Decodifica para OpenCV
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="invalid_image")
    
    # Extrai faces
    faces = model.get(img)
    
    if not faces:
        raise HTTPException(status_code=422, detail="no_face_detected")
    
    # Retorna o embedding da face mais destacada (maior bounding box)
    best_face = max(faces, key=lambda f: (
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ))
    
    return {
        "embedding":   best_face.embedding.tolist(),  # lista de 512 floats
        "confidence":  float(best_face.det_score),
        "faces_found": len(faces),
    }