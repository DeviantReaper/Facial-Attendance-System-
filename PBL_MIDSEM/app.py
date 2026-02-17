"""
NEURO-SCAN v3.0 - Facial Attendance System
==========================================
Features:
  - Face photo enrollment (students/ folder)
  - Real face recognition using MediaPipe landmarks + cosine similarity
  - Unknown face alert
  - CSV export
  - Live webcam enrollment via /enroll page
  - Dashboard with charts
  - Session reset
"""

import cv2
import mediapipe as mp
import numpy as np
import os, ssl, time, threading, json, csv, io, pickle
import urllib.request
from datetime import datetime
from collections import deque
from flask import Flask, Response, render_template, jsonify, request, send_file

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options

app = Flask(__name__)

# ── SSL fix (Python 3.13 macOS) ───────────────────────────────────────────────
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode    = ssl.CERT_NONE

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "students")
ENROLL_DB    = os.path.join(BASE_DIR, "enrollments.pkl")
os.makedirs(STUDENTS_DIR, exist_ok=True)

# ── Download models ───────────────────────────────────────────────────────────
MODELS = {
    "detector"   : ("blaze_face_short_range.tflite",
                    "https://storage.googleapis.com/mediapipe-models/face_detector/"
                    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"),
    "landmarker" : ("face_landmarker.task",
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                    "face_landmarker/float16/1/face_landmarker.task"),
}

def download_model(name, filename, url):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        print(f"  [setup] {name} model found.")
        return path
    print(f"  [setup] Downloading {name} model...")
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
    with opener.open(url) as r, open(path, "wb") as f:
        f.write(r.read())
    print(f"  [setup] {name} model saved -> {path}")
    return path

DETECTOR_PATH   = download_model("detector",   *MODELS["detector"])
LANDMARKER_PATH = download_model("landmarker", *MODELS["landmarker"])

# ── Build MediaPipe objects ───────────────────────────────────────────────────
# 1. FaceDetector — finds bounding boxes
det_opts = mp_vision.FaceDetectorOptions(
    base_options=mp_base_options.BaseOptions(model_asset_path=DETECTOR_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    min_detection_confidence=0.55,
)
face_detector = mp_vision.FaceDetector.create_from_options(det_opts)

# 2. FaceLandmarker — 478 3D landmarks used as face embedding
lm_opts = mp_vision.FaceLandmarkerOptions(
    base_options=mp_base_options.BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=6,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(lm_opts)
print("  [setup] FaceDetector + FaceLandmarker ready.")

# ── Face Enrollment Database ──────────────────────────────────────────────────
# Structure: { "Shreya": [ embedding_array, ... ], "Rahul": [...], ... }
enrollment_db = {}

def load_enrollment_db():
    global enrollment_db
    if os.path.exists(ENROLL_DB):
        with open(ENROLL_DB, "rb") as f:
            enrollment_db = pickle.load(f)
        print(f"  [enroll] Loaded {len(enrollment_db)} enrolled students.")
    else:
        enrollment_db = {}

def save_enrollment_db():
    with open(ENROLL_DB, "wb") as f:
        pickle.dump(enrollment_db, f)

def extract_landmark_embedding(bgr_frame):
    """Return a normalized 1434-dim numpy vector from face landmarks, or None."""
    rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = face_landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    # Take the first detected face's 478 landmarks (x, y, z)
    lms = result.face_landmarks[0]
    vec = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
    # Normalize: center around nose tip (landmark 1), divide by face width
    nose   = vec[1*3 : 1*3+3]
    vec_c  = vec.reshape(-1, 3) - nose
    scale  = np.linalg.norm(vec_c[33] - vec_c[263]) + 1e-6  # eye distance
    vec_c /= scale
    return vec_c.flatten()

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

RECOGNITION_THRESHOLD = 0.82   # tune: higher = stricter matching

def recognize_face(embedding):
    """Return (student_name, score) or ('UNKNOWN', 0.0)."""
    if not enrollment_db or embedding is None:
        return "UNKNOWN", 0.0
    best_name, best_score = "UNKNOWN", 0.0
    for name, embeddings in enrollment_db.items():
        for ref in embeddings:
            s = cosine_similarity(embedding, ref)
            if s > best_score:
                best_score = s
                best_name  = name if s >= RECOGNITION_THRESHOLD else "UNKNOWN"
    return best_name, best_score

# ── Load photos from students/ folder on startup ──────────────────────────────
def enroll_from_photos():
    """Scan students/ folder. Each file: StudentName.jpg (or .png)."""
    count = 0
    for fname in os.listdir(STUDENTS_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        name = os.path.splitext(fname)[0].replace("_", " ").strip()
        img  = cv2.imread(os.path.join(STUDENTS_DIR, fname))
        if img is None:
            print(f"  [enroll] Could not read {fname}, skipping.")
            continue
        emb = extract_landmark_embedding(img)
        if emb is None:
            print(f"  [enroll] No face found in {fname}, skipping.")
            continue
        if name not in enrollment_db:
            enrollment_db[name] = []
        enrollment_db[name].append(emb)
        count += 1
        print(f"  [enroll] Enrolled: {name}")
    save_enrollment_db()
    print(f"  [enroll] Total enrolled from photos: {count}")

load_enrollment_db()
enroll_from_photos()

# ── Session state ─────────────────────────────────────────────────────────────
attendance_log   = deque(maxlen=200)
marked_students  = {}          # name -> {time, confidence}
unknown_alerts   = deque(maxlen=20)
face_count_now   = 0
scan_active      = False
unknown_active   = False
session_start    = datetime.now()
last_log_time    = {}          # per-student cooldown
LOG_COOLDOWN     = 5

announce_queue   = deque(maxlen=10)
frame_lock       = threading.Lock()
latest_frame     = None
enroll_capture   = {"active": False, "name": "", "frames": [], "done": False, "msg": ""}


# ── Webcam thread ─────────────────────────────────────────────────────────────
def capture_frames():
    global latest_frame, face_count_now, scan_active, unknown_active

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("  [ERROR] Cannot open webcam. Check System Settings > Privacy > Camera.")
        return
    print("  [camera] Webcam opened.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det    = face_detector.detect(mp_img)
        faces  = det.detections or []

        has_unknown     = False
        faces_in_frame  = len(faces)

        for detection in faces:
            bbox  = detection.bounding_box
            x1    = max(0, bbox.origin_x)
            y1    = max(0, bbox.origin_y)
            x2    = min(w, x1 + bbox.width)
            y2    = min(h, y1 + bbox.height)
            score = detection.categories[0].score if detection.categories else 0.0

            # Get face embedding for recognition
            face_crop = frame[max(0,y1-20):min(h,y2+20), max(0,x1-20):min(w,x2+20)]
            embedding = extract_landmark_embedding(face_crop) if face_crop.size > 0 else None
            name, sim = recognize_face(embedding)

            # ── Enrollment capture mode ───────────────────────────────────
            if enroll_capture["active"] and embedding is not None:
                enroll_capture["frames"].append(embedding)
                if len(enroll_capture["frames"]) >= 10:
                    _finish_enrollment(enroll_capture["name"], enroll_capture["frames"])

            # ── Color + label based on recognition ────────────────────────
            if name == "UNKNOWN" or not enrollment_db:
                clr   = (0, 80, 255)   # red-orange for unknown
                label = f"UNKNOWN  {score:.2f}"
                has_unknown = True
            elif name in marked_students:
                clr   = (0, 220, 80)   # green = already marked
                label = f"{name}  MARKED"
            else:
                clr   = (0, 255, 200)  # cyan = recognized, not yet marked
                label = f"{name}  {sim:.2f}"

            # Corner bracket box
            c = 18
            for pts in [((x1,y1),(x1+c,y1)),((x1,y1),(x1,y1+c)),
                        ((x2,y1),(x2-c,y1)),((x2,y1),(x2,y1+c)),
                        ((x1,y2),(x1+c,y2)),((x1,y2),(x1,y2-c)),
                        ((x2,y2),(x2-c,y2)),((x2,y2),(x2,y2-c))]:
                cv2.line(frame, pts[0], pts[1], clr, 2)

            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)

            # Confidence bar
            bw   = x2 - x1
            fill = int(bw * score)
            cv2.rectangle(frame, (x1, y2+4), (x2, y2+8), (30,30,30), -1)
            cv2.rectangle(frame, (x1, y2+4), (x1+fill, y2+8), clr, -1)

            # ── Log attendance ────────────────────────────────────────────
            if name != "UNKNOWN" and enrollment_db:
                now = time.time()
                last = last_log_time.get(name, 0)
                if name not in marked_students and (now - last > LOG_COOLDOWN):
                    last_log_time[name] = now
                    marked_students[name] = {
                        "time"      : datetime.now().strftime("%H:%M:%S"),
                        "confidence": f"{sim:.1%}"
                    }
                    scan_active = True
                    attendance_log.appendleft({
                        "id"        : name,
                        "time"      : marked_students[name]["time"],
                        "confidence": f"{sim:.1%}",
                        "status"    : "PRESENT"
                    })
                    announce_queue.appendleft(f"{name} marked present")
                    threading.Timer(2.5, _clear_scan).start()

        # Unknown face alert
        if has_unknown and faces_in_frame > 0:
            if not unknown_active:
                unknown_active = True
                unknown_alerts.appendleft({
                    "time"   : datetime.now().strftime("%H:%M:%S"),
                    "message": "Unidentified face detected"
                })
                announce_queue.appendleft("Warning. Unidentified person detected.")
                threading.Timer(4.0, _clear_unknown).start()
        elif not has_unknown:
            unknown_active = False

        face_count_now = faces_in_frame
        _draw_hud(frame, w, h, faces_in_frame)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with frame_lock:
            latest_frame = buf.tobytes()

    cap.release()


def _finish_enrollment(name, embeddings):
    enroll_capture["active"] = False
    if name not in enrollment_db:
        enrollment_db[name] = []
    enrollment_db[name].extend(embeddings[:10])
    save_enrollment_db()
    enroll_capture["done"] = True
    enroll_capture["msg"]  = f"{name} enrolled with {len(embeddings)} samples."
    enroll_capture["frames"] = []
    announce_queue.appendleft(f"{name} successfully enrolled.")
    print(f"  [enroll] {name} enrolled via webcam.")


def _clear_scan():
    global scan_active
    scan_active = False

def _clear_unknown():
    global unknown_active
    unknown_active = False

def _elapsed():
    d = datetime.now() - session_start
    m, s = divmod(int(d.total_seconds()), 60)
    return f"{m:02d}:{s:02d}"

def _draw_hud(frame, w, h, n):
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,50), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"NEURO-SCAN v3.0  |  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
                (14,32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,200), 1, cv2.LINE_AA)
    clr = (0,255,100) if n else (80,80,80)
    cv2.putText(frame, f"FACES:{n}  MARKED:{len(marked_students)}",
                (w-310,32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0,h-40),(w,h),(0,0,0),-1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"PRESENT:{len(marked_students)}  ENROLLED:{len(enrollment_db)}  SESSION:{_elapsed()}",
                (14,h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,200,160), 1, cv2.LINE_AA)


# ── MJPEG ─────────────────────────────────────────────────────────────────────
def gen_frames():
    while True:
        with frame_lock:
            f = latest_frame
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.033)


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    announcement = announce_queue.pop() if announce_queue else None
    # Build chart data
    total_enrolled = len(enrollment_db)
    total_present  = len(marked_students)
    total_absent   = max(0, total_enrolled - total_present)
    per_student = [
        {"name": name, "time": info["time"], "confidence": info["confidence"]}
        for name, info in marked_students.items()
    ]
    return jsonify({
        "face_count"     : face_count_now,
        "scan_active"    : scan_active,
        "unknown_active" : unknown_active,
        "total_present"  : total_present,
        "total_enrolled" : total_enrolled,
        "total_absent"   : total_absent,
        "elapsed"        : _elapsed(),
        "log"            : list(attendance_log),
        "unknown_alerts" : list(unknown_alerts),
        "per_student"    : per_student,
        "announcement"   : announcement,
    })

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    enroll_capture["active"] = True
    enroll_capture["name"]   = name
    enroll_capture["frames"] = []
    enroll_capture["done"]   = False
    enroll_capture["msg"]    = f"Capturing {name}... look at the camera"
    return jsonify({"status": "capturing", "message": enroll_capture["msg"]})

@app.route("/api/enroll/status")
def api_enroll_status():
    return jsonify({
        "active"  : enroll_capture["active"],
        "done"    : enroll_capture["done"],
        "msg"     : enroll_capture["msg"],
        "captured": len(enroll_capture["frames"]),
        "needed"  : 10,
        "enrolled": list(enrollment_db.keys()),
    })

@app.route("/api/enroll/delete", methods=["POST"])
def api_enroll_delete():
    data = request.get_json()
    name = data.get("name", "")
    if name in enrollment_db:
        del enrollment_db[name]
        save_enrollment_db()
        return jsonify({"status": "deleted", "name": name})
    return jsonify({"error": "Not found"}), 404

@app.route("/api/reset", methods=["POST"])
def api_reset():
    global marked_students, attendance_log, session_start, last_log_time
    marked_students = {}
    attendance_log  = deque(maxlen=200)
    last_log_time   = {}
    session_start   = datetime.now()
    return jsonify({"status": "reset"})

@app.route("/api/export")
def api_export():
    """Download attendance as CSV."""
    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(["Student Name", "Time Marked", "Confidence", "Status", "Session Date"])
    date_str = session_start.strftime("%Y-%m-%d")
    for entry in reversed(list(attendance_log)):
        cw.writerow([entry["id"], entry["time"], entry["confidence"], entry["status"], date_str])
    output = io.BytesIO()
    output.write(si.getvalue().encode("utf-8"))
    output.seek(0)
    filename = f"attendance_{date_str}.csv"
    return send_file(output, mimetype="text/csv",
                     as_attachment=True, download_name=filename)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=capture_frames, daemon=True)
    t.start()
    print("\n  +==========================================+")
    print("  |  NEURO-SCAN v3.0  .  Attendance System  |")
    print("  |  Dashboard : http://127.0.0.1:8080      |")
    print("  |  Enroll    : http://127.0.0.1:8080/enroll |")
    print("  +==========================================+\n")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)