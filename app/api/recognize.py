import cv2, numpy as np, base64, uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta

from app.core.database import get_db, SessionLocal
from app.core.config import settings
from app.api.auth import require_teacher
from app.models.student import Student
from app.models.face_descriptor import FaceDescriptor
from app.models.attendance import AttendanceRecord
from app.services.face_service import face_detector
from app.services.embedding_service import embedding_service
from app.services.recognition_service import recognition_service

router = APIRouter()

# In-memory WS session registry  {session_id: [WebSocket, ...]}
active_sessions: dict[str, list[WebSocket]] = {}


@router.post("/recognize", summary="Recognize faces in image and mark attendance")
async def recognize_faces(
    file: UploadFile = File(...),
    class_id: Optional[int] = Query(None),
    session_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(require_teacher),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    detections = face_detector.detect_faces(img)
    if not detections:
        return {"detected": 0, "results": [], "message": "No faces found."}

    # Fetch all face descriptors for the class
    if class_id:
        from app.models.enrollment import Enrollment
        enrolled_ids = [
            e.student_id for e in db.query(Enrollment).filter(
                Enrollment.class_id == class_id, Enrollment.status == "active"
            ).all()
        ]
        descriptors = db.query(FaceDescriptor).filter(
            FaceDescriptor.student_id.in_(enrolled_ids)
        ).all()
    else:
        descriptors = db.query(FaceDescriptor).all()

    # Build student embedding map: {student_id: [emb1, emb2, ...]}
    emb_map: dict[int, list] = {}
    for desc in descriptors:
        emb_map.setdefault(desc.student_id, []).append(desc.embedding)

    id_to_student = {s.id: s for s in db.query(Student).filter(Student.is_active == True).all()}

    results = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        h, w = img.shape[:2]
        face_crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        emb = embedding_service.get_embedding(face_crop)
        if emb is None:
            results.append({"name": "Unknown", "confidence": det["confidence"], "box": det["box"]})
            continue

        match_id, similarity = recognition_service.match_face(emb, emb_map)

        name = "Unknown"
        attendance_logged = False
        student_id = None
        confidence = float(similarity) if similarity else 0.0

        if match_id and similarity >= (1 - settings.RECOGNITION_THRESHOLD):
            student = id_to_student.get(match_id)
            if student:
                name = student.name
                student_id = student.id
                # Auto-mark attendance if confidence passes threshold
                if confidence >= settings.ATTENDANCE_CONFIDENCE_THRESHOLD:
                    cooldown = datetime.utcnow() - timedelta(minutes=settings.ATTENDANCE_COOLDOWN_MINUTES)
                    existing = db.query(AttendanceRecord).filter(
                        AttendanceRecord.student_id == match_id,
                        AttendanceRecord.class_id == class_id,
                        AttendanceRecord.time_marked >= cooldown,
                    ).first() if class_id else None

                    if not existing:
                        record = AttendanceRecord(
                            student_id=match_id,
                            class_id=class_id,
                            status="present",
                            confidence_score=confidence,
                            session_id=session_id,
                        )
                        db.add(record)
                        db.commit()
                        attendance_logged = True

        result_item = {
            "name": name,
            "student_id": student_id,
            "confidence": round(confidence, 3),
            "box": det["box"],
            "attendance_logged": attendance_logged,
        }
        results.append(result_item)

        # Broadcast via WebSocket if session active
        if session_id and session_id in active_sessions and attendance_logged:
            import asyncio, json
            for ws in active_sessions.get(session_id, []):
                try:
                    asyncio.create_task(ws.send_text(json.dumps(result_item)))
                except Exception:
                    pass

    return {"detected": len(results), "results": results}


@router.websocket("/ws/attendance/{session_id}")
async def ws_attendance(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_sessions.setdefault(session_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        active_sessions.get(session_id, []).remove(websocket)
