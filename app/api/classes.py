from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import date

from app.core.database import get_db
from app.api.auth import require_teacher, get_current_user
from app.models.class_model import Class
from app.models.enrollment import Enrollment
from app.models.student import Student
from app.models.teacher import Teacher

router = APIRouter()


# ─── Schemas ────────────────────────────────────────────────────────────────

class ClassCreate(BaseModel):
    subject_name: str
    subject_code: str
    room_number: Optional[str] = None
    schedule: Optional[dict] = None
    color: Optional[str] = "#1e3a5f"
    min_attendance_pct: Optional[float] = 75.0
    late_threshold_minutes: Optional[int] = 10

class ClassUpdate(ClassCreate):
    pass

class EnrollStudentsRequest(BaseModel):
    student_ids: List[int]


# ─── Routes ─────────────────────────────────────────────────────────────────

@router.get("/classes", summary="List all classes for current teacher")
def list_classes(
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    teacher_id = int(current_user["sub"])
    classes = db.query(Class).filter(Class.teacher_id == teacher_id, Class.is_active == True).all()
    result = []
    for c in classes:
        enrolled_count = db.query(Enrollment).filter(
            Enrollment.class_id == c.id, Enrollment.status == "active"
        ).count()
        result.append({
            "id": c.id,
            "subject_name": c.subject_name,
            "subject_code": c.subject_code,
            "room_number": c.room_number,
            "schedule": c.schedule,
            "color": c.color,
            "min_attendance_pct": c.min_attendance_pct,
            "late_threshold_minutes": c.late_threshold_minutes,
            "enrolled_count": enrolled_count,
        })
    return result


@router.post("/classes", summary="Create a class")
def create_class(body: ClassCreate, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    teacher_id = int(current_user["sub"])
    cls = Class(teacher_id=teacher_id, **body.dict())
    db.add(cls)
    db.commit()
    db.refresh(cls)
    return {"id": cls.id, "subject_name": cls.subject_name, "subject_code": cls.subject_code}


@router.get("/classes/{class_id}", summary="Get class details")
def get_class(class_id: int, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    return cls


@router.put("/classes/{class_id}", summary="Update class")
def update_class(class_id: int, body: ClassUpdate, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    for k, v in body.dict(exclude_none=True).items():
        setattr(cls, k, v)
    db.commit()
    db.refresh(cls)
    return cls


@router.delete("/classes/{class_id}", summary="Delete (soft) class")
def delete_class(class_id: int, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    cls.is_active = False
    db.commit()
    return {"detail": "Class deleted"}


@router.get("/classes/{class_id}/students", summary="List students in a class")
def list_class_students(class_id: int, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    enrollments = db.query(Enrollment).filter(
        Enrollment.class_id == class_id, Enrollment.status == "active"
    ).all()
    students = []
    for e in enrollments:
        s = db.query(Student).filter(Student.id == e.student_id).first()
        if s:
            students.append({
                "id": s.id, "name": s.name, "roll_no": s.roll_no,
                "email": s.email, "is_enrolled": s.is_enrolled,
                "profile_photo_path": s.profile_photo_path,
            })
    return students


@router.post("/classes/{class_id}/students", summary="Enroll students in class")
def enroll_students(
    class_id: int,
    body: EnrollStudentsRequest,
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    added = []
    for sid in body.student_ids:
        existing = db.query(Enrollment).filter(
            Enrollment.class_id == class_id, Enrollment.student_id == sid
        ).first()
        if not existing:
            db.add(Enrollment(class_id=class_id, student_id=sid))
            added.append(sid)
    db.commit()
    return {"enrolled": added, "count": len(added)}


@router.delete("/classes/{class_id}/students/{student_id}", summary="Remove student from class")
def remove_student(class_id: int, student_id: int, db: Session = Depends(get_db), current_user=Depends(require_teacher)):
    e = db.query(Enrollment).filter(
        Enrollment.class_id == class_id, Enrollment.student_id == student_id
    ).first()
    if not e:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    e.status = "dropped"
    db.commit()
    return {"detail": "Student removed from class"}
