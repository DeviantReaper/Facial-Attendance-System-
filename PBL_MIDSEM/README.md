# NEURO-SCAN v3.0 · Facial Attendance System

## What's New in v3.0
- Real face recognition (MediaPipe 478-point landmarks + cosine similarity)
- Live webcam enrollment at http://127.0.0.1:8080/enroll
- Photo folder enrollment (drop .jpg files into students/)
- Unknown face red alert + voice warning
- Live donut chart (Present / Absent / Active Faces)
- CSV export button
- Session reset

---

## Folder Structure
```
facial_attendance_v2/
├── app.py
├── enrollments.pkl          ← auto-created, stores face embeddings
├── blaze_face_short_range.tflite   ← auto-downloaded
├── face_landmarker.task            ← auto-downloaded (~30 MB, first run)
├── students/                ← drop photos here for bulk enrollment
│   ├── Shreya_Sharma.jpg
│   ├── Rahul_Verma.jpg
│   └── ...
└── templates/
    ├── index.html           ← main dashboard
    └── enroll.html          ← enrollment page
```

---

## Setup & Run

```bash
# 1. Activate your existing venv
source .venv/bin/activate

# 2. No new packages needed (uses mediapipe + flask + opencv already installed)

# 3. Run
python app.py
```

First run downloads `face_landmarker.task` (~30 MB) automatically.

Open:
- Dashboard  → http://127.0.0.1:8080
- Enroll     → http://127.0.0.1:8080/enroll
- Export CSV → http://127.0.0.1:8080/api/export

---

## How to Add Students

### Method 1 — Webcam (recommended for demo)
1. Open http://127.0.0.1:8080/enroll
2. Type student's full name
3. Student looks at camera
4. Click CAPTURE & ENROLL
5. Done in ~3 seconds

### Method 2 — Photo file
1. Put a clear face photo in the `students/` folder
2. Name it `FirstName_LastName.jpg` (underscore becomes space)
3. Restart `python app.py`
4. The system auto-enrolls it on startup

---

## Tuning Recognition Accuracy

In `app.py` line ~85:
```python
RECOGNITION_THRESHOLD = 0.82
```
- Raise it (e.g. 0.88) → stricter, fewer false positives
- Lower it (e.g. 0.75) → more lenient, better in low light

---

## Troubleshooting

| Problem | Fix |
|---|---|
| face_landmarker.task download fails | Run: `curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"` |
| Student not recognized | Re-enroll in same lighting conditions |
| Everything shows UNKNOWN | No students enrolled yet — visit /enroll first |
| Black video feed | System Settings > Privacy > Camera > allow Terminal |