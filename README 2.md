# Face Attendance System — Backend

## Setup

```bash
# 1. Install dependencies (face_recognition needs cmake + dlib first on Windows)
pip install -r requirements.txt

# On Windows if dlib fails:
#   pip install cmake
#   pip install dlib
#   pip install face_recognition

# 2. Start the API server
uvicorn main:app --reload

# 3. Open interactive API docs
# http://127.0.0.1:8000/docs
```

---

## Endpoints

| Method | URL | What it does |
|--------|-----|-------------|
| POST | `/register/` | Register a face (name + photo) |
| POST | `/recognize/` | Detect + identify faces in photo |
| GET | `/users/` | List all registered users |
| DELETE | `/users/{name}` | Delete a user |
| POST | `/camera/start` | Open live webcam window |

---

## How to register your face (accuracy tip)

Register the **same person 3–5 times** using photos taken at:
- Different angles (slightly left, right, straight)
- Different lighting conditions

Each `POST /register/` call with the same name **adds** a new encoding.  
More encodings = better accuracy when recognizing.

```
POST /register/?name=Rahul
Body: form-data  →  file: photo1.jpg

POST /register/?name=Rahul
Body: form-data  →  file: photo2.jpg   ← different angle

GET /users/
→ {"users": [{"name": "Rahul", "encodings_count": 2}]}
```

---

## Live Camera

Either:
1. `POST /camera/start` via the API (opens window on server machine)
2. Or run directly: `python camera.py`

Press **ESC** to stop.

---

## Delete a face

```
DELETE /users/Rahul
→ {"message": "User 'Rahul' deleted successfully."}
```

---

## Accuracy Details

| Feature | This system | Basic ChatGPT version |
|---------|-------------|----------------------|
| Encodings per person | Multiple (add more anytime) | Single |
| Matching strategy | Voting + distance fallback | Single distance check |
| Strict threshold | 0.45 | 0.50 |
| Upsampling at registration | 2× | 1× |
| num_jitters at registration | 3 | default (1) |
| Multi-face support | ✅ | ✅ |

### Thresholds explained
- `0.45` → strict (high confidence) — used when multiple stored encodings agree
- `0.52` → loose fallback — used when only 1 encoding is stored or as tiebreaker
- Distance `< 0.6` is typically considered a match in research; we're more conservative

---

## Project Structure

```
face_attendance/
├── main.py          ← FastAPI app + all endpoints
├── database.py      ← SQLite connection
├── models.py        ← User table schema
├── face_utils.py    ← Encoding + voting recognition logic
├── camera.py        ← Standalone live camera script
├── requirements.txt
└── attendance.db    ← Created automatically on first run
```
