from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.security import verify_password, create_access_token, decode_token, hash_password, generate_api_key
from app.models.teacher import Teacher
from app.models.student import Student
from app.models.college import College

router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)


# ─── Schemas ────────────────────────────────────────────────────────────────

class TeacherLoginRequest(BaseModel):
    username: str
    password: str

class StudentLoginRequest(BaseModel):
    roll_no: str
    password: str

class CollegeTokenRequest(BaseModel):
    api_key: str
    college_code: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: int
    name: str


# ─── Dependency: get current user from JWT ───────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload  # {"sub": str(id), "role": "teacher"|"student"|"college", ...}


def require_teacher(current_user=Depends(get_current_user)):
    if current_user.get("role") not in ("teacher", "admin"):
        raise HTTPException(status_code=403, detail="Teacher access required")
    return current_user


def require_admin(current_user=Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ─── Routes ─────────────────────────────────────────────────────────────────

@router.post("/auth/login", response_model=TokenResponse, summary="Teacher login")
def teacher_login(body: TeacherLoginRequest, db: Session = Depends(get_db)):
    teacher = db.query(Teacher).filter(Teacher.username == body.username.strip().lower()).first()
    if not teacher or not verify_password(body.password, teacher.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not teacher.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    role = "admin" if teacher.is_admin else "teacher"
    token = create_access_token({"sub": str(teacher.id), "role": role, "name": teacher.name})
    return TokenResponse(access_token=token, role=role, user_id=teacher.id, name=teacher.name)


@router.post("/auth/student-login", response_model=TokenResponse, summary="Student login")
def student_login(body: StudentLoginRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.roll_no == body.roll_no.strip().upper()).first()
    if not student or not student.password_hash or not verify_password(body.password, student.password_hash):
        raise HTTPException(status_code=401, detail="Invalid roll number or password")
    if not student.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token({"sub": str(student.id), "role": "student", "name": student.name})
    return TokenResponse(access_token=token, role="student", user_id=student.id, name=student.name)


@router.post("/auth/college-token", response_model=TokenResponse, summary="Partner college API token")
def college_token(body: CollegeTokenRequest, db: Session = Depends(get_db)):
    college = db.query(College).filter(College.code == body.college_code.strip().upper()).first()
    if not college or not college.is_active:
        raise HTTPException(status_code=401, detail="Invalid college code")
    if not verify_password(body.api_key, college.api_key_hash):
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = create_access_token({"sub": str(college.id), "role": "college", "name": college.name})
    return TokenResponse(access_token=token, role="college", user_id=college.id, name=college.name)


@router.get("/auth/me", summary="Get current user info")
def get_me(current_user=Depends(get_current_user)):
    return current_user
