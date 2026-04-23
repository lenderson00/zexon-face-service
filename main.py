from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import insightface
import numpy as np
import cv2

app = FastAPI(title="Zexon Face Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo uma vez ao subir o serviço
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1)  # -1 = CPU (Railway não tem GPU)

@app.get("/health")
def health():
    return { "status": "ok", "model": "buffalo_l" }

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