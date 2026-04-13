# FaceAttend — Face Recognition Attendance Portal

> Production-grade, AI-powered attendance management system for universities and colleges.

![Portal Design](./docs/preview.png)

---

## ✨ Features

### 🎓 Teacher / Admin Portal
- **Secure JWT login** with role-based access (teacher vs admin)
- **Dashboard** → Today's attendance, stat cards, weekly bar chart, activity feed
- **Class Management** → Create/edit/delete classes with schedule, room, color coding
- **Student Management** → Add manually, bulk CSV import, face enrollment status
- **Face Enrollment** → Upload 3–5 photos OR auto-capture 5 webcam frames; quality scoring
- **Live Attendance** → Start session, real-time face recognition via webcam with bounding box overlays, confidence badges (green/yellow/red), manual override, session summary
- **Reports** → Daily bar chart, weekly trend line, pie chart, defaulters list with progress bars, PDF/Excel export
- **Settings** → Institution config, attendance rules (min %, late threshold, cooldown), email notifications, appearance
- **Integrations** → Partner college management, API key generation, webhook config

### 👨‍🎓 Student Portal
- **Login** with roll number + password
- **Dashboard** → Per-subject donut charts, at-risk alerts, 30-day calendar heatmap
- **My Subjects** → Full breakdown with pie chart, present/late/absent counts
- **Attendance History** → Filterable table with confidence scores, certificate download
- **Profile** → Edit personal info, change password
- **Self-Enrollment** → 4-step wizard: personal details → 8-angle webcam capture → quality check → review & submit

### 🤝 College Integration API
```
POST   /api/v1/auth/college-token        Get JWT via API key
POST   /api/v1/integrate/enroll          Enroll external student + face
GET    /api/v1/integrate/students        List external students
POST   /api/v1/integrate/attendance      Log attendance
GET    /api/v1/integrate/attendance/:id  Get attendance record
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18 + Vite + react-router-dom v6 |
| **UI** | Vanilla CSS (navy `#1e3a5f` + orange `#f97316`) · Inter font |
| **Charts** | Recharts (bar, line, pie, donut) |
| **Face (browser)** | `@vladmandic/face-api` · SSD MobileNet v1 |
| **Backend** | FastAPI (Python 3.11+) |
| **Face (server)** | DeepFace · FaceNet512 · cosine similarity |
| **Detection** | YOLOv8n (server) · SSD MobileNet (browser) |
| **Database** | SQLite (dev) / PostgreSQL (prod) · SQLAlchemy |
| **Auth** | JWT (python-jose) · bcrypt (passlib) |
| **Real-time** | WebSocket (native FastAPI) |
| **Notifications** | SMTP email (optional) |
| **Containers** | Docker + Docker Compose |

---

## 🚀 Quick Start (Local Dev)

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Backend Setup

```bash
cd Facial-Attendance-System--main

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your settings

# Seed the database
python scripts/seed_data.py

# Start the backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: **http://localhost:8000/docs**

### 2. Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# Download face-api.js model weights
cd .. && bash scripts/download_models.sh && cd frontend

# Copy env
echo "VITE_API_URL=http://localhost:8000/api/v1" > .env

# Start dev server
npm run dev
```

App available at: **http://localhost:5173**

### 3. Demo Login

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Teacher | `prof.sharma` | `professor123` |
| Teacher | `prof.iyer` | `professor123` |
| Student | `MUJ2022001` | `student123` |
| Student | `MUJ2022002` | `student123` |

---

## 🐳 Docker Compose (Full Stack)

```bash
# Build and start all services (PostgreSQL + Redis + Backend + Frontend nginx)
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
Facial-Attendance-System--main/
├── app/
│   ├── api/
│   │   ├── auth.py          # JWT login (teacher/student/college)
│   │   ├── classes.py       # Class CRUD + student enrollment
│   │   ├── students.py      # Student CRUD + bulk CSV import
│   │   ├── register.py      # Face enrollment + quality check
│   │   ├── recognize.py     # Face recognition + WS broadcast
│   │   ├── attendance.py    # Records, reports, defaulters
│   │   └── integrate.py     # College integration API
│   ├── core/
│   │   ├── config.py        # Settings (pydantic)
│   │   ├── database.py      # SQLAlchemy engine
│   │   └── security.py      # JWT + bcrypt + API key
│   ├── models/
│   │   ├── student.py       # Student table
│   │   ├── teacher.py       # Teacher table
│   │   ├── class_model.py   # Class table
│   │   ├── enrollment.py    # Student-class junction
│   │   ├── face_descriptor.py # 512-d embeddings
│   │   ├── attendance.py    # Attendance records
│   │   ├── college.py       # Partner colleges
│   │   └── notification.py  # Notifications
│   └── services/
│       ├── face_service.py       # YOLOv8 face detection
│       ├── embedding_service.py  # DeepFace FaceNet512
│       ├── recognition_service.py# Cosine similarity matching
│       ├── quality_service.py    # Blur/brightness checks
│       └── attendance_service.py # Cooldown-aware marking
├── frontend/
│   ├── public/models/       # face-api.js model weights
│   └── src/
│       ├── pages/
│       │   ├── LoginPage.jsx
│       │   ├── Dashboard.jsx
│       │   ├── Classes.jsx
│       │   ├── ClassDetail.jsx   # Live attendance camera
│       │   ├── Students.jsx
│       │   ├── FaceEnrollment.jsx
│       │   ├── Reports.jsx
│       │   ├── Settings.jsx
│       │   ├── Integrations.jsx
│       │   ├── History.jsx
│       │   └── student/
│       │       ├── StudentDashboard.jsx
│       │       ├── StudentSubjects.jsx
│       │       ├── StudentHistory.jsx
│       │       ├── StudentEnrollment.jsx
│       │       └── StudentProfile.jsx
│       ├── components/
│       │   ├── Layout.jsx
│       │   └── StudentLayout.jsx
│       ├── context/
│       │   ├── AuthContext.jsx
│       │   └── ThemeContext.jsx
│       └── services/api.js   # Axios + JWT interceptors
├── scripts/
│   ├── seed_data.py          # Sample data seeder
│   └── download_models.sh    # face-api.js model downloader
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── Dockerfile
```

---

## 🔑 Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | SQLite (dev) or PostgreSQL connection string |
| `JWT_SECRET` | Random secret for JWT signing (min 32 chars) |
| `RECOGNITION_MODEL` | DeepFace model (`Facenet512` recommended) |
| `RECOGNITION_THRESHOLD` | Cosine distance threshold (0.4 = strict) |
| `ATTENDANCE_CONFIDENCE_THRESHOLD` | Min confidence to auto-mark (0.75) |
| `EMAILS_ENABLED` | Set `true` to enable SMTP notifications |

---

## 📡 API Documentation

Full Swagger UI available at: **http://localhost:8000/docs**

Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | Teacher login → JWT |
| POST | `/api/v1/auth/student-login` | Student login → JWT |
| GET  | `/api/v1/classes` | List teacher's classes |
| POST | `/api/v1/register` | Enroll student face |
| POST | `/api/v1/recognize` | Recognize + mark attendance |
| GET  | `/api/v1/attendance` | Filter attendance records |
| GET  | `/api/v1/attendance/reports/defaulters` | Defaulters list |
| POST | `/api/v1/integrate/colleges` | Create partner college + API key |

---

## 🎯 Face Recognition Pipeline

```
Input frame
    ↓
[Browser] SSD MobileNet v1 (face-api.js)   — fast detection
    ↓
[Server]  DeepFace FaceNet512              — 512-d embedding
    ↓
[Server]  Cosine similarity vs stored descriptors
    ↓
Confidence ≥ 0.75?  →  Mark Present
             < 0.75?  →  Unknown / Manual review
```

---

## 📦 Deployment

See `DEPLOY.md` for detailed guides for:
- **VPS** (Ubuntu + nginx + gunicorn + PostgreSQL)
- **Railway** (one-click backend + DB)
- **Vercel** (frontend)
- **Render** (backend)

---

## 📝 License

MIT License — for academic and personal use.

---

*Built with ❤️ for Manipal University Jaipur*