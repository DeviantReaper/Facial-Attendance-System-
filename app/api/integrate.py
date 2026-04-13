from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import cv2, numpy as np

from app.core.database import get_db
from app.core.security import hash_password, generate_api_key, verify_password
from app.api.auth import get_current_user
from app.models.college import College
from app.models.student import Student
from app.models.face_descriptor import FaceDescriptor
from app.models.attendance import AttendanceRecord
from app.services.face_service import face_detector
from app.services.embedding_service import embedding_service
from app.services.quality_service import quality_service

router = APIRouter()


class CollegeCreate(BaseModel):
    name: str
    code: str
    contact_email: Optional[str] = None
    contact_name: Optional[str] = None
    webhook_url: Optional[str] = None
    auto_approve_enrollment: bool = False


class ExternalStudentEnroll(BaseModel):
    name: str
    roll_no: str
    email: str
    phone: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[int] = None


class ExternalAttendanceLog(BaseModel):
    student_roll_no: str
    class_id: int
    status: str = "present"
    confidence_score: Optional[float] = None


# ─── College Management (Admin only) ─────────────────────────────────────────

@router.get("/integrate/colleges", summary="List partner colleges")
def list_colleges(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    colleges = db.query(College).filter(College.is_active == True).all()
    return [
        {"id": c.id, "name": c.name, "code": c.code, "contact_email": c.contact_email,
         "auto_approve_enrollment": c.auto_approve_enrollment}
        for c in colleges
    ]


@router.post("/integrate/colleges", summary="Register a new partner college and generate API key")
def create_college(body: CollegeCreate, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    existing = db.query(College).filter(College.code == body.code.upper()).first()
    if existing:
        raise HTTPException(status_code=409, detail="College code already exists")

    raw_key = generate_api_key()
    college = College(
        name=body.name,
        code=body.code.upper(),
        api_key_hash=hash_password(raw_key),
        contact_email=body.contact_email,
        contact_name=body.contact_name,
        webhook_url=body.webhook_url,
        auto_approve_enrollment=body.auto_approve_enrollment,
    )
    db.add(college)
    db.commit()
    db.refresh(college)
    # Return raw key ONCE — never stored in plain text
    return {"college_id": college.id, "code": college.code, "api_key": raw_key,
            "warning": "Save this API key — it will not be shown again."}


@router.post("/integrate/colleges/{college_id}/regenerate-key", summary="Regenerate API key")
def regenerate_key(college_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    college = db.query(College).filter(College.id == college_id).first()
    if not college:
        raise HTTPException(status_code=404, detail="College not found")
    raw_key = generate_api_key()
    college.api_key_hash = hash_password(raw_key)
    db.commit()
    return {"api_key": raw_key, "warning": "Save this API key — it will not be shown again."}


# ─── External Enrollment API ─────────────────────────────────────────────────

@router.post("/integrate/enroll", summary="Enroll external student + face")
async def external_enroll(
    name: str = Form(...),
    roll_no: str = Form(...),
    email: str = Form(...),
    phone: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    semester: Optional[int] = Form(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    college_id = int(current_user["sub"]) if current_user.get("role") == "college" else None

    # Find or create student
    student = db.query(Student).filter(Student.roll_no == roll_no.upper()).first()
    if not student:
        student = Student(
            name=name, roll_no=roll_no.upper(), email=email,
            phone=phone, department=department, semester=semester,
            password_hash=hash_password("student123"),
            college_id=college_id,
            enrollment_status="approved" if college_id else "pending",
        )
        db.add(student)
        db.commit()
        db.refresh(student)

    # Process face photos
    enrolled = 0
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        quality = quality_service.check_quality(img)
        detections = face_detector.detect_faces(img)
        if not detections:
            continue
        det = detections[0]
        x1, y1, x2, y2 = det["box"]
        h, w = img.shape[:2]
        face_crop = img[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
        emb = embedding_service.get_embedding(face_crop)
        if emb:
            db.add(FaceDescriptor(
                student_id=student.id, embedding=emb,
                quality_score=quality["overall_score"],
                blur_score=quality["blur_score"],
                brightness_score=quality["brightness_score"],
            ))
            enrolled += 1

    if enrolled > 0:
        student.is_enrolled = True
    db.commit()

    return {"student_id": student.id, "roll_no": student.roll_no,
            "faces_enrolled": enrolled, "status": student.enrollment_status}


@router.get("/integrate/students", summary="Get all external students for this college")
def get_external_students(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    college_id = int(current_user["sub"])
    students = db.query(Student).filter(Student.college_id == college_id, Student.is_active == True).all()
    return [
        {"id": s.id, "name": s.name, "roll_no": s.roll_no, "email": s.email,
         "is_enrolled": s.is_enrolled, "enrollment_status": s.enrollment_status}
        for s in students
    ]


@router.post("/integrate/attendance", summary="Log attendance for external student")
def log_external_attendance(body: ExternalAttendanceLog, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    student = db.query(Student).filter(Student.roll_no == body.student_roll_no.upper()).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    record = AttendanceRecord(
        student_id=student.id,
        class_id=body.class_id,
        status=body.status,
        confidence_score=body.confidence_score,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "status": record.status}


@router.get("/integrate/attendance/{student_roll_no}", summary="Get attendance for external student")
def get_external_attendance(student_roll_no: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    student = db.query(Student).filter(Student.roll_no == student_roll_no.upper()).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    records = db.query(AttendanceRecord).filter(AttendanceRecord.student_id == student.id).all()
    return [
        {"id": r.id, "class_id": r.class_id, "date": r.date.isoformat() if r.date else None,
         "status": r.status, "confidence_score": r.confidence_score}
        for r in records
    ]
