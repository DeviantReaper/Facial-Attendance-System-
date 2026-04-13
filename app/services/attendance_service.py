from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.attendance import Attendance
from app.core.config import settings

class AttendanceService:
    def log_attendance(self, db: Session, user_id: int):
        """
        Log attendance for a user if they haven't been logged recently.
        """
        # Check for recent logs (cooldown)
        cooldown_time = datetime.utcnow() - timedelta(minutes=settings.ATTENDANCE_COOLDOWN_MINUTES)
        
        recent_log = db.query(Attendance).filter(
            Attendance.user_id == user_id,
            Attendance.timestamp >= cooldown_time
        ).first()

        if not recent_log:
            new_log = Attendance(user_id=user_id, timestamp=datetime.utcnow())
            db.add(new_log)
            db.commit()
            db.refresh(new_log)
            return new_log
        
        return None # Already logged recently

# Singleton instance
attendance_service = AttendanceService()
