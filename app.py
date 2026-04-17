import os
os.environ["INSIGHTFACE_HOME"] = "/app"

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import cv2
import psycopg2
from pgvector.psycopg2 import register_vector
import requests
from dotenv import load_dotenv
import boto3
from io import BytesIO
from PIL import Image

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_KEY        = os.getenv("AI_API_KEY")
DATABASE_URL   = os.getenv("DATABASE_URL")
THRESHOLD      = float(os.getenv("SIMILARITY_THRESHOLD", 65))  # ← default 65 (skala 0-100)
R2_ACCESS_KEY  = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY  = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT    = os.getenv("R2_ENDPOINT_URL")
R2_BUCKET      = os.getenv("R2_BUCKET_NAME")

print("INSIGHTFACE_HOME =", os.getenv("INSIGHTFACE_HOME"))

# ─────────────────────────────────────────
# INIT FASTAPI
# ─────────────────────────────────────────
app = FastAPI(
    title="KeJepret AI Service",
    description="Face recognition service untuk platform foto event lari",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# INIT INSIGHTFACE — model dari image, tidak download!
# ─────────────────────────────────────────
print("⏳ Loading InsightFace model...")
face_app = FaceAnalysis(
    name="buffalo_l",
    root="/app",   # InsightFace akan baca /app/models/buffalo_l/
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace model siap!")

# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runner_embeddings (
            id SERIAL PRIMARY KEY,
            runner_id INTEGER UNIQUE NOT NULL,
            embedding vector(512) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS photo_embeddings (
            id SERIAL PRIMARY KEY,
            photo_id INTEGER NOT NULL,
            face_index INTEGER NOT NULL,
            embedding vector(512) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_photo_embeddings_photo_id
        ON photo_embeddings(photo_id);
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database tables siap!")

# ─────────────────────────────────────────
# R2 CLIENT
# ─────────────────────────────────────────
r2_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto"
)

# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def load_image_from_url(url: str) -> np.ndarray:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Gagal decode gambar")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal load gambar: {str(e)}")

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar")
    return img

def get_best_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    faces = face_app.get(img)
    if not faces:
        return None
    best_face = max(faces, key=lambda f: (
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ))
    return best_face.embedding

def get_all_embeddings(img: np.ndarray) -> List[np.ndarray]:
    faces = face_app.get(img)
    return [face.embedding for face in faces]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # ← FIX: normalisasi dulu seperti simulasi, bukan langsung dot/norm
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    return float(np.dot(norm_a, norm_b))  # hasil: -1 ~ 1

# ─────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────
class EnrollRequest(BaseModel):
    runner_id: int
    selfie_url: str

class EnrollResponse(BaseModel):
    status: str
    runner_id: int
    message: str

class EmbedPhotoRequest(BaseModel):
    photo_id: int
    photo_url: str

class EmbedPhotoResponse(BaseModel):
    status: str
    photo_id: int
    faces_found: int
    message: str

class SearchRequest(BaseModel):
    runner_id: int
    photo_ids: List[int]

class PhotoMatch(BaseModel):
    photo_id: int
    score: float

class SearchResponse(BaseModel):
    status: str
    runner_id: int
    matched: List[PhotoMatch]
    total_scanned: int
    total_matched: int

class HealthResponse(BaseModel):
    status: str
    model: str
    threshold: float
    version: str

# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="online",
        model="InsightFace buffalo_l",
        threshold=THRESHOLD,
        version="1.0.0"
    )

@app.post("/enroll", response_model=EnrollResponse)
def enroll_runner(
    request: EnrollRequest,
    api_key: str = Depends(verify_api_key),
    db=Depends(get_db)
):
    img = load_image_from_url(request.selfie_url)
    embedding = get_best_embedding(img)
    if embedding is None:
        raise HTTPException(status_code=422, detail="Tidak ada wajah terdeteksi.")
    cur = db.cursor()
    cur.execute("""
        INSERT INTO runner_embeddings (runner_id, embedding, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (runner_id)
        DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW();
    """, (request.runner_id, embedding.tolist()))
    db.commit()
    cur.close()
    return EnrollResponse(
        status="success",
        runner_id=request.runner_id,
        message=f"Wajah runner {request.runner_id} berhasil didaftarkan"
    )

@app.post("/embed-photo", response_model=EmbedPhotoResponse)
def embed_photo(
    request: EmbedPhotoRequest,
    api_key: str = Depends(verify_api_key),
    db=Depends(get_db)
):
    img = load_image_from_url(request.photo_url)
    embeddings = get_all_embeddings(img)
    faces_found = len(embeddings)
    if faces_found == 0:
        return EmbedPhotoResponse(
            status="no_face",
            photo_id=request.photo_id,
            faces_found=0,
            message="Tidak ada wajah terdeteksi dalam foto"
        )
    cur = db.cursor()
    cur.execute("DELETE FROM photo_embeddings WHERE photo_id = %s", (request.photo_id,))
    for idx, embedding in enumerate(embeddings):
        cur.execute("""
            INSERT INTO photo_embeddings (photo_id, face_index, embedding)
            VALUES (%s, %s, %s)
        """, (request.photo_id, idx, embedding.tolist()))
    db.commit()
    cur.close()
    return EmbedPhotoResponse(
        status="success",
        photo_id=request.photo_id,
        faces_found=faces_found,
        message=f"Berhasil embed {faces_found} wajah dari foto {request.photo_id}"
    )

@app.post("/search", response_model=SearchResponse)
def search_runner_photos(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key),
    db=Depends(get_db)
):
    cur = db.cursor()
    cur.execute(
        "SELECT embedding FROM runner_embeddings WHERE runner_id = %s",
        (request.runner_id,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Runner {request.runner_id} belum terdaftar.")
    runner_embedding = np.array(row[0])
    matched = []
    for photo_id in request.photo_ids:
        cur.execute(
            "SELECT embedding FROM photo_embeddings WHERE photo_id = %s",
            (photo_id,)
        )
        rows = cur.fetchall()
        if not rows:
            continue
        best_score = 0.0
        for row in rows:
            face_embedding = np.array(row[0])
            raw_score = cosine_similarity(runner_embedding, face_embedding)
            # ← FIX: konversi -1~1 ke 0-100% seperti simulasi
            persen = (raw_score + 1) / 2 * 100
            if persen > best_score:
                best_score = persen
        # ← FIX: bandingkan persen (0-100) dengan THRESHOLD (65)
        if best_score >= THRESHOLD:
            matched.append(PhotoMatch(photo_id=photo_id, score=round(best_score, 1)))
    cur.close()
    matched.sort(key=lambda x: x.score, reverse=True)
    return SearchResponse(
        status="success",
        runner_id=request.runner_id,
        matched=matched,
        total_scanned=len(request.photo_ids),
        total_matched=len(matched)
    )

@app.delete("/runner/{runner_id}")
def delete_runner_embedding(runner_id: int, api_key: str = Depends(verify_api_key), db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("DELETE FROM runner_embeddings WHERE runner_id = %s", (runner_id,))
    db.commit()
    cur.close()
    return {"status": "success", "message": f"Embedding runner {runner_id} dihapus"}

@app.delete("/photo/{photo_id}")
def delete_photo_embeddings(photo_id: int, api_key: str = Depends(verify_api_key), db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("DELETE FROM photo_embeddings WHERE photo_id = %s", (photo_id,))
    db.commit()
    cur.close()
    return {"status": "success", "message": f"Embedding foto {photo_id} dihapus"}