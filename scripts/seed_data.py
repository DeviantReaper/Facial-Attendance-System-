"""
Seed script — populates the database with sample teachers, students, classes,
enrollments, face descriptors, and attendance records for testing.

Usage:
    python scripts/seed_data.py

Requirements: Backend dependencies must be installed.
"""
import sys, os, random, json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal, Base, engine
from app.core.security import hash_password
from app.models.teacher import Teacher
from app.models.student import Student
from app.models.class_model import Class
from app.models.enrollment import Enrollment
from app.models.face_descriptor import FaceDescriptor
from app.models.attendance import AttendanceRecord
from app.models.college import College

Base.metadata.create_all(bind=engine)

db = SessionLocal()

def seed():
    print("🌱 Seeding database…")

    # ── Colleges ──────────────────────────────────────────────────────────────
    colleges_data = [
        {"name": "Jaipur Institute of Technology", "code": "JIT", "contact_email": "admin@jit.edu"},
        {"name": "Rajasthan Technical University",  "code": "RTU", "contact_email": "admin@rtu.ac.in"},
    ]
    colleges = {}
    for c in colleges_data:
        existing = db.query(College).filter(College.code == c["code"]).first()
        if not existing:
            col = College(
                name=c["name"], code=c["code"],
                contact_email=c["contact_email"],
                api_key_hash=hash_password("demo_api_key"),
                auto_approve_enrollment=False,
            )
            db.add(col)
            db.flush()
            colleges[c["code"]] = col
            print(f"  ✓ College: {c['name']}")
        else:
            colleges[c["code"]] = existing

    # ── Teachers ───────────────────────────────────────────────────────────────
    teachers_data = [
        {"name": "Prof. Rajesh Sharma",  "username": "prof.sharma",  "email": "sharma@muj.edu",  "is_admin": True},
        {"name": "Prof. Kavita Iyer",    "username": "prof.iyer",    "email": "iyer@muj.edu",    "is_admin": False},
        {"name": "Prof. Amit Verma",     "username": "prof.verma",   "email": "verma@muj.edu",   "is_admin": False},
    ]
    teachers = {}
    for t in teachers_data:
        existing = db.query(Teacher).filter(Teacher.username == t["username"]).first()
        if not existing:
            teacher = Teacher(
                name=t["name"], username=t["username"], email=t["email"],
                password_hash=hash_password("professor123"),
                institution="Manipal University Jaipur", department="Computer Science",
                is_admin=t["is_admin"],
            )
            db.add(teacher)
            db.flush()
            teachers[t["username"]] = teacher
            print(f"  ✓ Teacher: {t['name']} (pw: professor123)")
        else:
            teachers[t["username"]] = existing

    # ── Students ───────────────────────────────────────────────────────────────
    students_data = [
        {"name": "Arjun Mehta",    "roll_no": "MUJ2022001", "email": "arjun@muj.edu",   "dept": "CSE", "sem": 6},
        {"name": "Priya Singh",    "roll_no": "MUJ2022002", "email": "priya@muj.edu",   "dept": "CSE", "sem": 6},
        {"name": "Rohit Kumar",    "roll_no": "MUJ2022003", "email": "rohit@muj.edu",   "dept": "ECE", "sem": 4},
        {"name": "Sneha Gupta",    "roll_no": "MUJ2022004", "email": "sneha@muj.edu",   "dept": "CSE", "sem": 6},
        {"name": "Vikas Sharma",   "roll_no": "MUJ2022005", "email": "vikas@muj.edu",   "dept": "ME",  "sem": 2},
        {"name": "Ananya Patel",   "roll_no": "MUJ2022006", "email": "ananya@muj.edu",  "dept": "CSE", "sem": 4},
        {"name": "Karan Joshi",    "roll_no": "MUJ2022007", "email": "karan@muj.edu",   "dept": "EE",  "sem": 6},
        {"name": "Divya Yadav",    "roll_no": "MUJ2022008", "email": "divya@muj.edu",   "dept": "CSE", "sem": 4},
        {"name": "Mohit Agarwal",  "roll_no": "MUJ2022009", "email": "mohit@muj.edu",   "dept": "CE",  "sem": 2},
        {"name": "Ritu Sharma",    "roll_no": "MUJ2022010", "email": "ritu@muj.edu",    "dept": "CSE", "sem": 8},
    ]
    students = []
    for s in students_data:
        existing = db.query(Student).filter(Student.roll_no == s["roll_no"]).first()
        if not existing:
            student = Student(
                name=s["name"], roll_no=s["roll_no"], email=s["email"],
                department=s["dept"], semester=s["sem"],
                password_hash=hash_password("student123"),
                is_enrolled=True, enrollment_status="approved",
            )
            db.add(student)
            db.flush()
            students.append(student)
            print(f"  ✓ Student: {s['name']} ({s['roll_no']}) pw: student123")
        else:
            students.append(existing)

    # ── Classes ────────────────────────────────────────────────────────────────
    teacher = list(teachers.values())[0]
    classes_data = [
        {"name": "Machine Learning",  "code": "CS601", "room": "LHC-3", "color": "#f97316", "days": "Mon/Wed", "time": "10:00"},
        {"name": "Database Systems",  "code": "CS401", "room": "LHC-1", "color": "#1e3a5f", "days": "Tue/Thu", "time": "14:00"},
        {"name": "Computer Networks", "code": "CS501", "room": "LHC-2", "color": "#10b981", "days": "Mon/Fri", "time": "09:00"},
    ]
    classes = []
    for c in classes_data:
        existing = db.query(Class).filter(Class.subject_code == c["code"]).first()
        if not existing:
            cls = Class(
                subject_name=c["name"], subject_code=c["code"],
                room_number=c["room"], color=c["color"],
                teacher_id=teacher.id, min_attendance_pct=75,
                schedule={"day": c["days"], "time": c["time"], "duration": 60},
            )
            db.add(cls)
            db.flush()
            classes.append(cls)
            print(f"  ✓ Class: {c['name']} ({c['code']})")
        else:
            classes.append(existing)

    # ── Enrollments ────────────────────────────────────────────────────────────
    for student in students:
        for cls in classes:
            existing = db.query(Enrollment).filter(
                Enrollment.student_id == student.id, Enrollment.class_id == cls.id
            ).first()
            if not existing:
                db.add(Enrollment(student_id=student.id, class_id=cls.id, status="active"))
    print(f"  ✓ Enrollments created")

    # ── Mock face descriptors (512-d random vectors) ───────────────────────────
    for student in students[:7]:  # First 7 students get faces enrolled
        existing = db.query(FaceDescriptor).filter(FaceDescriptor.student_id == student.id).first()
        if not existing:
            for _ in range(3):
                emb = [random.gauss(0, 0.5) for _ in range(512)]
                db.add(FaceDescriptor(
                    student_id=student.id,
                    embedding=emb,
                    quality_score=round(0.75 + random.random() * 0.2, 3),
                    blur_score=round(100 + random.random() * 200, 1),
                    brightness_score=round(100 + random.random() * 80, 1),
                ))
    print(f"  ✓ Face descriptors added (7 students)")

    # ── Attendance records (last 30 days) ─────────────────────────────────────
    existing_records = db.query(AttendanceRecord).count()
    if existing_records == 0:
        for cls in classes:
            for i in range(15):  # 15 session days
                session_date = datetime.utcnow() - timedelta(days=i * 2)
                for student in students:
                    r = random.random()
                    status = "present" if r > 0.2 else ("late" if r > 0.1 else "absent")
                    db.add(AttendanceRecord(
                        student_id=student.id, class_id=cls.id,
                        date=session_date, status=status,
                        confidence_score=round(0.78 + random.random() * 0.2, 3),
                        session_id=f"seed_sess_{cls.id}_{i}",
                    ))
        print(f"  ✓ Attendance records seeded (15 sessions × 3 classes × {len(students)} students)")

    db.commit()
    print("\n✅ Database seeded successfully!")
    print("\nDemo credentials:")
    print("  Teachers   — pw: professor123")
    print("    prof.sharma (admin)  ·  prof.iyer  ·  prof.verma")
    print("  Students   — pw: student123")
    print("    MUJ2022001 … MUJ2022010")

if __name__ == "__main__":
    seed()
    db.close()
