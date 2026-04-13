"""
Face Recognition Attendance System - Backend
============================================
Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs

Endpoints:
  POST   /register/         - Register a face (name + image file)
  POST   /recognize/        - Recognize face(s) in an uploaded image
  GET    /users/            - List all registered users
  DELETE /users/{name}      - Delete a user by name
  POST   /camera/start      - Start real-time camera (runs locally)
"""

import pickle
import face_recognition
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, engine
from models import User, Base
from face_utils import get_face_encodings_robust, compare_faces_vote

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Face Attendance API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────────
@app.post("/register/")
async def register(name: str, file: UploadFile = File(...)):
    """
    Register a person's face.
    - Supply their name and a clear photo (front-facing, good lighting).
    - Multiple uploads of the SAME name will ADD more encodings → better accuracy.
    """
    db = SessionLocal()
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        encodings = get_face_encodings_robust(image)

        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected. Use a clear, front-facing photo.")

        if len(encodings) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple faces ({len(encodings)}) detected. Use a photo with only one person for registration."
            )

        encoding = encodings[0]

        # Check if user already exists → append encoding for better accuracy
        existing = db.query(User).filter(User.name == name).first()
        if existing:
            stored: list = pickle.loads(existing.encoding)
            stored.append(encoding)
            existing.encoding = pickle.dumps(stored)
            db.commit()
            return {"message": f"Additional encoding added for '{name}'. Total: {len(stored)} encodings."}
        else:
            user = User(name=name, encoding=pickle.dumps([encoding]))
            db.add(user)
            db.commit()
            return {"message": f"'{name}' registered successfully with 1 encoding."}
    finally:
        db.close()


# ─────────────────────────────────────────────
# RECOGNIZE (multi-face support)
# ─────────────────────────────────────────────
@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    """
    Recognize all faces in an uploaded image.
    Returns name for each detected face (or 'Unknown').
    """
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            return {"faces_detected": 0, "results": [], "message": "No registered users yet."}

        # Build lookup: name → list of encodings
        name_encodings: dict[str, list] = {}
        for user in users:
            name_encodings[user.name] = pickle.loads(user.encoding)

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use 'cnn' model for better accuracy if GPU available, else 'hog'
        face_locations = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(rgb, face_locations, num_jitters=2)

        results = []
        for enc in face_encodings:
            name = compare_faces_vote(name_encodings, enc)
            results.append(name)

        return {
            "faces_detected": len(results),
            "results": results
        }
    finally:
        db.close()


# ─────────────────────────────────────────────
# LIST USERS
# ─────────────────────────────────────────────
@app.get("/users/")
def list_users():
    """List all registered users and how many face encodings each has."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        result = []
        for u in users:
            encodings = pickle.loads(u.encoding)
            result.append({"name": u.name, "encodings_count": len(encodings)})
        return {"users": result, "total": len(result)}
    finally:
        db.close()


# ─────────────────────────────────────────────
# DELETE USER
# ─────────────────────────────────────────────
@app.delete("/users/{name}")
def delete_user(name: str):
    """Delete a registered user by name."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.name == name).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{name}' not found.")
        db.delete(user)
        db.commit()
        return {"message": f"User '{name}' deleted successfully."}
    finally:
        db.close()


# ─────────────────────────────────────────────
# CAMERA (local real-time, run directly)
# ─────────────────────────────────────────────
@app.post("/camera/start")
def start_camera():
    """Trigger real-time camera window on the server machine. Press ESC to stop."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            return {"message": "No registered users. Please register faces first."}

        name_encodings: dict[str, list] = {
            u.name: pickle.loads(u.encoding) for u in users
        }
    finally:
        db.close()

    run_camera(name_encodings)
    return {"message": "Camera session ended."}


def run_camera(name_encodings: dict):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("Camera started. Press ESC to quit.")
    process_every_n = 3   # process every 3rd frame for speed
    frame_count = 0
    last_results = []     # cache results to display on skipped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        if frame_count % process_every_n == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small, model="hog", number_of_times_to_upsample=1)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations, num_jitters=1)

            last_results = []
            for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
                # Scale back to original frame size
                top *= 2; right *= 2; bottom *= 2; left *= 2
                name = compare_faces_vote(name_encodings, enc)
                last_results.append((top, right, bottom, left, name))

        for (top, right, bottom, left, name) in last_results:
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(display_frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
            cv2.putText(display_frame, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        cv2.putText(display_frame, f"Faces: {len(last_results)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.imshow("Face Attendance (ESC to quit)", display_frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
