"""
camera.py — Standalone real-time face recognition
==================================================
Run this directly (no server needed):

    python camera.py

Press ESC to quit.

This reads the same SQLite database the API uses, so you can:
  1. Register faces via the API  (POST /register/)
  2. Then run this script for live recognition
"""

import pickle
import face_recognition
import numpy as np
import cv2
from database import SessionLocal, engine
from models import User, Base
from face_utils import compare_faces_vote

Base.metadata.create_all(bind=engine)


def load_known_faces() -> dict[str, list]:
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return {u.name: pickle.loads(u.encoding) for u in users}
    finally:
        db.close()


def run():
    name_encodings = load_known_faces()

    if not name_encodings:
        print("⚠  No registered faces found. Register via POST /register/ first.")
        return

    print(f"✅ Loaded {len(name_encodings)} user(s): {list(name_encodings.keys())}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # Optional: try to set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera started. Press ESC to quit.")

    process_every_n = 3
    frame_count = 0
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        if frame_count % process_every_n == 0:
            # Shrink for speed, then scale locations back
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(
                rgb_small, model="hog", number_of_times_to_upsample=1
            )
            encodings = face_recognition.face_encodings(rgb_small, locations, num_jitters=1)

            last_results = []
            for (top, right, bottom, left), enc in zip(locations, encodings):
                top    *= 2
                right  *= 2
                bottom *= 2
                left   *= 2
                name = compare_faces_vote(name_encodings, enc)
                last_results.append((top, right, bottom, left, name))

        for (top, right, bottom, left, name) in last_results:
            color = (0, 210, 0) if name != "Unknown" else (0, 0, 220)
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            # Label background
            cv2.rectangle(display, (left, bottom - 32), (right, bottom), color, cv2.FILLED)
            cv2.putText(display, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1)

        cv2.putText(display, f"Tracking {len(last_results)} face(s)  |  ESC = quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow("Face Attendance - Live", display)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    run()
