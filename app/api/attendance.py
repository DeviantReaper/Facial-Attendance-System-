from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime, timedelta

from app.core.database import get_db
from app.api.auth import require_teacher, get_current_user
from app.models.attendance import AttendanceRecord
from app.models.student import Student
from app.models.class_model import Class
from app.models.enrollment import Enrollment

router = APIRouter()


class AttendanceOverride(BaseModel):
    status: str       # present, absent, late
    note: Optional[str] = None


@router.get("/attendance", summary="List attendance records with filters")
def list_attendance(
    class_id: Optional[int] = Query(None),
    student_id: Optional[int] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = db.query(AttendanceRecord)
    if class_id:
        q = q.filter(AttendanceRecord.class_id == class_id)
    if student_id:
        q = q.filter(AttendanceRecord.student_id == student_id)
    if start_date:
        q = q.filter(AttendanceRecord.date >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        q = q.filter(AttendanceRecord.date <= datetime.combine(end_date, datetime.max.time()))
    if status:
        q = q.filter(AttendanceRecord.status == status)

    records = q.order_by(AttendanceRecord.date.desc()).limit(limit).all()
    result = []
    for r in records:
        student = db.query(Student).filter(Student.id == r.student_id).first()
        cls = db.query(Class).filter(Class.id == r.class_id).first()
        result.append({
            "id": r.id,
            "student_id": r.student_id,
            "student_name": student.name if student else "Unknown",
            "roll_no": student.roll_no if student else "",
            "class_id": r.class_id,
            "class_name": cls.subject_name if cls else "Unknown",
            "subject_code": cls.subject_code if cls else "",
            "date": r.date.isoformat() if r.date else None,
            "status": r.status,
            "confidence_score": r.confidence_score,
            "is_manual_override": r.is_manual_override,
            "time_marked": r.time_marked.isoformat() if r.time_marked else None,
        })
    return result


@router.put("/attendance/{record_id}", summary="Manual override attendance record")
def override_attendance(
    record_id: int,
    body: AttendanceOverride,
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    record = db.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    record.status = body.status
    record.is_manual_override = True
    record.marked_by = int(current_user["sub"])
    db.commit()
    return {"id": record.id, "status": record.status}


@router.get("/attendance/reports/summary", summary="Attendance summary per class")
def attendance_summary(
    class_id: Optional[int] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = db.query(AttendanceRecord)
    if class_id:
        q = q.filter(AttendanceRecord.class_id == class_id)
    if start_date:
        q = q.filter(AttendanceRecord.date >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        q = q.filter(AttendanceRecord.date <= datetime.combine(end_date, datetime.max.time()))

    records = q.all()
    total = len(records)
    present = sum(1 for r in records if r.status == "present")
    absent = sum(1 for r in records if r.status == "absent")
    late = sum(1 for r in records if r.status == "late")

    return {
        "total": total,
        "present": present,
        "absent": absent,
        "late": late,
        "rate": round((present / total * 100), 1) if total else 0,
    }


@router.get("/attendance/reports/defaulters", summary="Students below minimum attendance threshold")
def defaulters_list(
    class_id: int = Query(...),
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    enrolled = db.query(Enrollment).filter(
        Enrollment.class_id == class_id, Enrollment.status == "active"
    ).all()

    defaulters = []
    for e in enrolled:
        total = db.query(AttendanceRecord).filter(
            AttendanceRecord.student_id == e.student_id,
            AttendanceRecord.class_id == class_id,
        ).count()
        present = db.query(AttendanceRecord).filter(
            AttendanceRecord.student_id == e.student_id,
            AttendanceRecord.class_id == class_id,
            AttendanceRecord.status == "present",
        ).count()
        pct = round((present / total * 100), 1) if total else 0
        if pct < cls.min_attendance_pct:
            s = db.query(Student).filter(Student.id == e.student_id).first()
            defaulters.append({
                "student_id": e.student_id,
                "name": s.name if s else "Unknown",
                "roll_no": s.roll_no if s else "",
                "attendance_pct": pct,
                "present": present,
                "total": total,
                "minimum_required": cls.min_attendance_pct,
            })
    return sorted(defaulters, key=lambda x: x["attendance_pct"])


@router.get("/attendance/student/{student_id}/stats", summary="Per-student stats per class")
def student_stats(
    student_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    enrollments = db.query(Enrollment).filter(
        Enrollment.student_id == student_id, Enrollment.status == "active"
    ).all()

    result = []
    for e in enrollments:
        cls = db.query(Class).filter(Class.id == e.class_id).first()
        records = db.query(AttendanceRecord).filter(
            AttendanceRecord.student_id == student_id,
            AttendanceRecord.class_id == e.class_id,
        ).all()
        total = len(records)
        present = sum(1 for r in records if r.status == "present")
        late = sum(1 for r in records if r.status == "late")
        absent = sum(1 for r in records if r.status == "absent")
        pct = round(((present + late) / total * 100), 1) if total else 0
        result.append({
            "class_id": e.class_id,
            "subject_name": cls.subject_name if cls else "",
            "subject_code": cls.subject_code if cls else "",
            "color": cls.color if cls else "#1e3a5f",
            "total": total,
            "present": present,
            "late": late,
            "absent": absent,
            "attendance_pct": pct,
            "min_required": cls.min_attendance_pct if cls else 75,
            "is_at_risk": pct < (cls.min_attendance_pct if cls else 75),
        })
    return result
