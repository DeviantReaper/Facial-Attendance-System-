import os, csv, io
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List

from app.core.database import get_db
from app.core.security import hash_password
from app.api.auth import require_teacher, get_current_user
from app.models.student import Student
from app.models.face_descriptor import FaceDescriptor
from app.models.attendance import AttendanceRecord
from app.models.class_model import Class

router = APIRouter()


class StudentCreate(BaseModel):
    name: str
    roll_no: str
    email: str
    phone: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[int] = None
    password: Optional[str] = "student123"


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[int] = None


@router.get("/students", summary="List all students")
def list_students(
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    q = db.query(Student).filter(Student.is_active == True)
    if search:
        q = q.filter(
            Student.name.ilike(f"%{search}%") |
            Student.roll_no.ilike(f"%{search}%") |
            Student.email.ilike(f"%{search}%")
        )
    total = q.count()
    students = q.offset((page - 1) * limit).limit(limit).all()
    result = []
    for s in students:
        face_count = db.query(FaceDescriptor).filter(FaceDescriptor.student_id == s.id).count()
        result.append({
            "id": s.id, "name": s.name, "roll_no": s.roll_no,
            "email": s.email, "phone": s.phone,
            "department": s.department, "semester": s.semester,
            "is_enrolled": s.is_enrolled,
            "enrollment_status": s.enrollment_status,
            "profile_photo_path": s.profile_photo_path,
            "face_count": face_count,
        })
    return {"total": total, "students": result}


@router.post("/students", summary="Create a new student")
def create_student(body: StudentCreate, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    existing = db.query(Student).filter(Student.roll_no == body.roll_no.upper()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Student with this roll number already exists")
    student = Student(
        name=body.name,
        roll_no=body.roll_no.upper(),
        email=body.email,
        phone=body.phone,
        department=body.department,
        semester=body.semester,
        password_hash=hash_password(body.password or "student123"),
        enrollment_status="approved",
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return {"id": student.id, "name": student.name, "roll_no": student.roll_no}


@router.get("/students/{student_id}", summary="Get student details")
def get_student(student_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    s = db.query(Student).filter(Student.id == student_id, Student.is_active == True).first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    face_count = db.query(FaceDescriptor).filter(FaceDescriptor.student_id == s.id).count()
    return {
        "id": s.id, "name": s.name, "roll_no": s.roll_no,
        "email": s.email, "phone": s.phone,
        "department": s.department, "semester": s.semester,
        "is_enrolled": s.is_enrolled, "enrollment_status": s.enrollment_status,
        "profile_photo_path": s.profile_photo_path, "face_count": face_count,
        "college_id": s.college_id,
    }


@router.put("/students/{student_id}", summary="Update student")
def update_student(student_id: int, body: StudentUpdate, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    s = db.query(Student).filter(Student.id == student_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    for k, v in body.dict(exclude_none=True).items():
        setattr(s, k, v)
    db.commit()
    db.refresh(s)
    return {"id": s.id, "name": s.name}


@router.delete("/students/{student_id}", summary="Soft-delete student")
def delete_student(student_id: int, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    s = db.query(Student).filter(Student.id == student_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Student not found")
    s.is_active = False
    db.commit()
    return {"detail": "Student deleted"}


@router.post("/students/bulk-import", summary="Bulk import students via CSV")
async def bulk_import(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    created = []
    errors = []
    for row in reader:
        try:
            roll_no = row.get("roll_no", "").strip().upper()
            if not roll_no:
                continue
            existing = db.query(Student).filter(Student.roll_no == roll_no).first()
            if existing:
                errors.append(f"{roll_no}: already exists")
                continue
            s = Student(
                name=row.get("name", "").strip(),
                roll_no=roll_no,
                email=row.get("email", "").strip(),
                phone=row.get("phone", "").strip() or None,
                department=row.get("department", "").strip() or None,
                semester=int(row["semester"]) if row.get("semester") else None,
                password_hash=hash_password("student123"),
                enrollment_status="approved",
            )
            db.add(s)
            created.append(roll_no)
        except Exception as exc:
            errors.append(f"Row error: {exc}")
    db.commit()
    return {"created": len(created), "errors": errors}


@router.get("/students/{student_id}/attendance", summary="Get student attendance history")
def student_attendance(
    student_id: int,
    class_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = db.query(AttendanceRecord).filter(AttendanceRecord.student_id == student_id)
    if class_id:
        q = q.filter(AttendanceRecord.class_id == class_id)
    records = q.order_by(AttendanceRecord.date.desc()).limit(200).all()
    return [
        {
            "id": r.id,
            "class_id": r.class_id,
            "date": r.date.isoformat() if r.date else None,
            "status": r.status,
            "confidence_score": r.confidence_score,
            "is_manual_override": r.is_manual_override,
            "time_marked": r.time_marked.isoformat() if r.time_marked else None,
        }
        for r in records
    ]
