import cv2, numpy as np
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.api.auth import require_teacher, get_current_user
from app.models.student import Student
from app.models.face_descriptor import FaceDescriptor
from app.services.face_service import face_detector
from app.services.embedding_service import embedding_service
from app.services.quality_service import quality_service

router = APIRouter()


@router.post("/register", summary="Enroll student face (upload or webcam frames)")
async def register_face(
    student_id: int = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    student = db.query(Student).filter(Student.id == student_id, Student.is_active == True).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    enrolled_count = 0
    quality_scores = []
    errors = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            errors.append(f"{file.filename}: invalid image")
            continue

        # Quality check
        quality = quality_service.check_quality(img)
        if quality["blur_score"] < 50.0:
            errors.append(f"{file.filename}: image too blurry (score={quality['blur_score']:.1f})")
            continue

        # Detect face
        detections = face_detector.detect_faces(img)
        if not detections:
            errors.append(f"{file.filename}: no face detected")
            continue

        # Take the largest face
        det = max(detections, key=lambda x: (x["box"][2]-x["box"][0]) * (x["box"][3]-x["box"][1]))
        x1, y1, x2, y2 = det["box"]
        h, w = img.shape[:2]
        pad = 20
        face_crop = img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]

        # Get embedding
        emb = embedding_service.get_embedding(face_crop)
        if emb is None:
            errors.append(f"{file.filename}: could not extract face embedding")
            continue

        # Store descriptor
        descriptor = FaceDescriptor(
            student_id=student_id,
            embedding=emb,
            quality_score=quality["overall_score"],
            blur_score=quality["blur_score"],
            brightness_score=quality["brightness_score"],
        )
        db.add(descriptor)
        quality_scores.append(quality["overall_score"])
        enrolled_count += 1

    if enrolled_count > 0:
        student.is_enrolled = True
        student.enrollment_status = "approved"
        db.commit()

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    return {
        "enrolled": enrolled_count,
        "errors": errors,
        "avg_quality_score": round(avg_quality, 3),
        "student_id": student_id,
        "student_name": student.name,
    }


@router.delete("/register/{student_id}", summary="Remove all face data for a student")
def clear_face_data(student_id: int, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    count = db.query(FaceDescriptor).filter(FaceDescriptor.student_id == student_id).delete()
    student = db.query(Student).filter(Student.id == student_id).first()
    if student:
        student.is_enrolled = False
    db.commit()
    return {"removed": count}


@router.get("/register/{student_id}/quality", summary="Get face descriptor quality scores")
def get_quality(student_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    descriptors = db.query(FaceDescriptor).filter(FaceDescriptor.student_id == student_id).all()
    return [
        {
            "id": d.id,
            "quality_score": d.quality_score,
            "blur_score": d.blur_score,
            "brightness_score": d.brightness_score,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in descriptors
    ]
