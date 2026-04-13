import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.api import register, recognize, users, attendance
from app.api import auth, classes, students, integrate
from app.core.config import settings
from app.core.database import Base, engine

# ── Import ALL models so SQLAlchemy relationship registry is complete ─────────
from app.models.teacher import Teacher          # noqa: F401
from app.models.student import Student          # noqa: F401
from app.models.class_model import Class        # noqa: F401
from app.models.enrollment import Enrollment    # noqa: F401
from app.models.face_descriptor import FaceDescriptor  # noqa: F401
from app.models.attendance import AttendanceRecord     # noqa: F401
from app.models.college import College          # noqa: F401
from app.models.notification import Notification       # noqa: F401

# Create all tables
Base.metadata.create_all(bind=engine)

# Ensure upload directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs("./data/photos", exist_ok=True)

app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    description="Production-grade Face Recognition Attendance Portal",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth.router,      prefix=settings.API_V1_STR, tags=["Auth"])
app.include_router(register.router,  prefix=settings.API_V1_STR, tags=["Face Enrollment"])
app.include_router(recognize.router, prefix=settings.API_V1_STR, tags=["Recognition"])
app.include_router(attendance.router,prefix=settings.API_V1_STR, tags=["Attendance"])
app.include_router(classes.router,   prefix=settings.API_V1_STR, tags=["Classes"])
app.include_router(students.router,  prefix=settings.API_V1_STR, tags=["Students"])
app.include_router(integrate.router, prefix=settings.API_V1_STR, tags=["College Integration"])
# Legacy routers
app.include_router(users.router,     prefix=settings.API_V1_STR, tags=["Users (Legacy)"])

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs",
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
