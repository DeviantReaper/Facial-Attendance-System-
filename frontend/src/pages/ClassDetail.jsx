import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Camera, CameraOff, UserCheck, UserX, ArrowLeft, Eye,
  CheckCircle, AlertTriangle, XCircle, Clock, Download, Users
} from "lucide-react";
import toast from "react-hot-toast";
import { classesAPI, attendanceAPI, studentsAPI } from "../services/api";

/* ── Mock data ─────────────────────────────────────────────────────────────── */
const MOCK_CLASS = {
  id: 1, subject_name: "Machine Learning", subject_code: "CS601",
  room_number: "LHC-3", enrolled_count: 42, color: "#f97316",
  schedule: { day: "Mon/Wed", time: "10:00 AM" }, min_attendance_pct: 75,
};

const MOCK_ENROLLED = [
  { id: 1, name: "Arjun Mehta",    roll_no: "MUJ2022001", is_enrolled: true },
  { id: 2, name: "Priya Singh",    roll_no: "MUJ2022002", is_enrolled: true },
  { id: 3, name: "Rohit Kumar",    roll_no: "MUJ2022003", is_enrolled: false },
  { id: 4, name: "Sneha Gupta",    roll_no: "MUJ2022004", is_enrolled: true },
  { id: 5, name: "Vikas Sharma",   roll_no: "MUJ2022005", is_enrolled: true },
  { id: 6, name: "Ananya Patel",   roll_no: "MUJ2022006", is_enrolled: false },
];

function StatusBadge({ status }) {
  if (status === "present") return <span className="badge badge-success">✓ Present</span>;
  if (status === "late")    return <span className="badge badge-warning">⏱ Late</span>;
  if (status === "absent")  return <span className="badge badge-error">✗ Absent</span>;
  return <span className="badge badge-gray">—</span>;
}

export default function ClassDetail() {
  const { classId } = useParams();
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const [cls, setCls] = useState(MOCK_CLASS);
  const [students, setStudents] = useState(MOCK_ENROLLED);
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionId] = useState(() => `sess_${Date.now()}`);
  const [attendance, setAttendance] = useState({});   // {student_id: "present"|"absent"|"late"}
  const [detections, setDetections] = useState([]);   // current frame results
  const [showSummary, setShowSummary] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // Enrollment modal state
  const [showEnrollModal, setShowEnrollModal] = useState(false);
  const [availableStudents, setAvailableStudents] = useState([]);
  const [selectedToEnroll, setSelectedToEnroll] = useState([]);
  const [enrolling, setEnrolling] = useState(false);

  useEffect(() => {
    classesAPI.get(classId).then((r) => setCls(r.data)).catch(() => {});
    fetchClassStudents();
  }, [classId]);

  const fetchClassStudents = () => {
    classesAPI.listStudents(classId).then((r) => { if (r.data?.length) setStudents(r.data); }).catch(() => {});
  };

  const openEnrollModal = async () => {
    try {
      const res = await studentsAPI.list();
      const allStudents = res.data.students || res.data || [];
      // Filter out those already in the `students` state
      const enrolledIds = new Set(students.map((s) => s.id));
      const available = allStudents.filter((s) => !enrolledIds.has(s.id));
      setAvailableStudents(available);
      setSelectedToEnroll([]);
      setShowEnrollModal(true);
    } catch {
      toast.error("Failed to load students");
    }
  };

  const handleEnroll = async () => {
    if (selectedToEnroll.length === 0) return toast.error("Select at least one student");
    setEnrolling(true);
    try {
      await classesAPI.enrollStudents(classId, { student_ids: selectedToEnroll });
      toast.success(`Enrolled ${selectedToEnroll.length} student(s)`);
      setShowEnrollModal(false);
      fetchClassStudents();
    } catch {
      toast.error("Failed to enroll students");
    }
    setEnrolling(false);
  };


  // ── Camera helpers ──────────────────────────────────────────────────────────
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" } });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch {
      toast.error("Could not access webcam");
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    clearInterval(intervalRef.current);
  };

  // ── Session control ─────────────────────────────────────────────────────────
  const startSession = async () => {
    setSessionActive(true);
    setAttendance({});
    setDetections([]);
    await startCamera();
    // Poll every 2 seconds
    intervalRef.current = setInterval(captureAndRecognize, 2000);
    toast.success("Attendance session started");
  };

  const stopSession = () => {
    setSessionActive(false);
    stopCamera();
    setShowSummary(true);
  };

  useEffect(() => () => stopCamera(), []);

  // ── Recognition ─────────────────────────────────────────────────────────────
  const captureAndRecognize = useCallback(async () => {
    if (!videoRef.current?.srcObject) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);

    // Draw face overlays on visible canvas
    const ovCtx = canvasRef.current?.getContext("2d");
    if (canvasRef.current && ovCtx) {
      canvasRef.current.width  = canvas.width;
      canvasRef.current.height = canvas.height;
    }

    canvas.toBlob(async (blob) => {
      const fd = new FormData();
      fd.append("file", blob, "frame.jpg");
      try {
        const res = await attendanceAPI.recognize(fd, { class_id: classId, session_id: sessionId });
        const results = res.data.results || [];
        setDetections(results);

        results.forEach((r) => {
          if (r.student_id && r.confidence >= 0.75) {
            setAttendance((prev) => {
              if (prev[r.student_id]) return prev;  // already marked
              return { ...prev, [r.student_id]: r.confidence >= 0.75 ? "present" : "late" };
            });
          }
        });

        // Draw bounding boxes
        if (ovCtx && canvasRef.current) {
          ovCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          const scaleX = canvasRef.current.offsetWidth / canvas.width;
          const scaleY = canvasRef.current.offsetHeight / canvas.height;
          results.forEach((r) => {
            const [x1, y1, x2, y2] = r.box;
            const conf = r.confidence;
            const color = conf >= 0.9 ? "#10b981" : conf >= 0.7 ? "#f59e0b" : "#ef4444";
            ovCtx.strokeStyle = color;
            ovCtx.lineWidth = 2;
            ovCtx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
            // Label
            ovCtx.fillStyle = color;
            ovCtx.fillRect(x1 * scaleX, (y1 * scaleY) - 22, 160, 22);
            ovCtx.fillStyle = "#fff";
            ovCtx.font = "bold 12px Inter, sans-serif";
            ovCtx.fillText(`${r.name}  ${(conf * 100).toFixed(0)}%`, x1 * scaleX + 5, y1 * scaleY - 6);
          });
        }
      } catch {
        // Simulate mock detections for demo
        const mockResult = MOCK_ENROLLED.filter((s) => s.is_enrolled).slice(0, 2).map((s) => ({
          name: s.name, student_id: s.id, confidence: 0.88 + Math.random() * 0.1, box: [100, 80, 300, 350],
        }));
        setDetections(mockResult);
        mockResult.forEach((r) => {
          if (r.student_id) setAttendance((prev) => prev[r.student_id] ? prev : { ...prev, [r.student_id]: "present" });
        });
      }
    }, "image/jpeg", 0.9);
  }, [classId, sessionId]);

  const manualToggle = (studentId) => {
    setAttendance((prev) => {
      const cur = prev[studentId];
      if (!cur)          return { ...prev, [studentId]: "present" };
      if (cur === "present") return { ...prev, [studentId]: "late" };
      if (cur === "late")    return { ...prev, [studentId]: "absent" };
      return { ...prev, [studentId]: "present" };
    });
  };

  /* ── Summary ── */
  const present = Object.values(attendance).filter((v) => v === "present").length;
  const late    = Object.values(attendance).filter((v) => v === "late").length;
  const absent  = students.length - present - late;

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate("/classes")}>
            <ArrowLeft size={16} /> Back
          </button>
          <div style={{ width: 4, height: 40, borderRadius: 4, background: cls.color, flexShrink: 0 }} />
          <div>
            <h2 style={{ fontSize: "1.3rem", fontWeight: 800 }}>
              {cls.subject_code} — {cls.subject_name}
            </h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginTop: 2 }}>
              {cls.room_number && `Room ${cls.room_number} · `}
              {cls.schedule?.day} {cls.schedule?.time} · {students.length} students
            </p>
          </div>
        </div>
        {!sessionActive ? (
          <button className="btn btn-primary" onClick={startSession}>
            <Camera size={16}/> Start Attendance
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopSession}>
            <CameraOff size={16}/> End Session
          </button>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: "1.25rem" }}>
        {/* ── Camera feed ── */}
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "1rem 1.25rem", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ fontWeight: 700 }}>Live Camera Feed</div>
            {sessionActive && (
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#ef4444", animation: "pulse 1.5s infinite" }} />
                <span style={{ fontSize: "0.78rem", color: "var(--error)", fontWeight: 700 }}>LIVE</span>
                <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>· {detections.length} face(s) detected</span>
              </div>
            )}
          </div>
          <div className="camera-container" style={{ maxWidth: "100%", margin: 0, borderRadius: 0, aspectRatio: "16/9" }}>
            <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            <canvas ref={canvasRef} className="camera-canvas" />
            {!sessionActive && (
              <div style={{
                position: "absolute", inset: 0, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)", gap: "1rem",
              }}>
                <Camera size={48} color="rgba(255,255,255,0.3)" />
                <p style={{ color: "rgba(255,255,255,0.6)", fontSize: "0.9rem" }}>
                  Click "Start Attendance" to begin face recognition
                </p>
              </div>
            )}
          </div>
          {/* Detection chips */}
          {sessionActive && detections.length > 0 && (
            <div style={{ padding: "0.875rem 1.25rem", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              {detections.map((d, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: "0.35rem",
                  padding: "3px 10px", borderRadius: "var(--radius-full)", fontSize: "0.75rem", fontWeight: 700,
                  background: d.confidence >= 0.9 ? "var(--success-bg)" : d.confidence >= 0.7 ? "var(--warning-bg)" : "var(--error-bg)",
                  color: d.confidence >= 0.9 ? "var(--success)" : d.confidence >= 0.7 ? "var(--warning)" : "var(--error)",
                }}>
                  {d.name} · {(d.confidence * 100).toFixed(0)}%
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Student attendance list ── */}
        <div className="card" style={{ padding: 0, display: "flex", flexDirection: "column", maxHeight: 520 }}>
          <div style={{ padding: "1rem 1.25rem", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
            <div style={{ fontWeight: 700 }}>Student Roster</div>
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              <div style={{ display: "flex", gap: "0.5rem", fontSize: "0.75rem", fontWeight: 700, marginRight: "0.5rem" }}>
                <span className="badge badge-success">{present}P</span>
                <span className="badge badge-warning">{late}L</span>
                <span className="badge badge-error">{absent}A</span>
              </div>
              <button className="btn btn-outline btn-sm" onClick={openEnrollModal}><UserCheck size={14}/> Add Students</button>
            </div>
          </div>
          <div style={{ overflowY: "auto", flex: 1 }}>
            {students.map((s) => {
              const status = attendance[s.id];
              return (
                <div key={s.id} style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "0.75rem 1.25rem", borderBottom: "1px solid var(--border)",
                  transition: "var(--transition)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                    <div style={{
                      width: 34, height: 34, borderRadius: "50%",
                      background: status === "present" ? "var(--success-bg)" : status === "late" ? "var(--warning-bg)" : "var(--border)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: status === "present" ? "var(--success)" : status === "late" ? "var(--warning)" : "var(--text-light)",
                      fontWeight: 700, fontSize: "0.78rem",
                    }}>
                      {s.name[0]}
                    </div>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: "0.85rem" }}>{s.name}</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{s.roll_no}</div>
                    </div>
                  </div>
                  <button onClick={() => manualToggle(s.id)} style={{ background: "none", border: "none", cursor: "pointer" }}>
                    <StatusBadge status={status} />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Summary Modal ── */}
      {showSummary && (
        <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && setShowSummary(false)}>
          <div className="modal animate-slide-up">
            <div className="modal-header">
              <div className="modal-title">Session Summary</div>
            </div>
            <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "1rem", textAlign: "center" }}>
                {[["Present", present, "success"], ["Late", late, "warning"], ["Absent", absent, "error"]].map(([l, v, t]) => (
                  <div key={l} style={{ padding: "1.25rem", borderRadius: "var(--radius)", background: `var(--${t}-bg)` }}>
                    <div style={{ fontSize: "2rem", fontWeight: 800, color: `var(--${t})` }}>{v}</div>
                    <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>{l}</div>
                  </div>
                ))}
              </div>
              <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", textAlign: "center" }}>
                Total students: <strong>{students.length}</strong> · Attendance rate:{" "}
                <strong style={{ color: "var(--primary)" }}>
                  {Math.round(((present + late) / (students.length || 1)) * 100)}%
                </strong>
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-outline" onClick={() => setShowSummary(false)}>Close</button>
              <button className="btn btn-primary" onClick={() => { setShowSummary(false); toast.success("Attendance saved!"); }}>
                <CheckCircle size={15}/> Save & Finish
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Enroll Students Modal ── */}
      {showEnrollModal && (
        <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && setShowEnrollModal(false)}>
          <div className="modal animate-slide-up" style={{ width: 500, maxWidth: "90%" }}>
            <div className="modal-header">
              <div className="modal-title">Enroll Students</div>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowEnrollModal(false)}><XCircle size={18}/></button>
            </div>
            
            <div className="modal-body" style={{ maxHeight: 400, overflowY: "auto", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {availableStudents.length === 0 ? (
                <div style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
                  All available students are already enrolled in this class.
                </div>
              ) : (
                availableStudents.map((stu) => (
                  <label key={stu.id} style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.75rem", background: "var(--bg-card)", borderRadius: "var(--radius)", cursor: "pointer", border: "1px solid var(--border)" }}>
                    <input 
                      type="checkbox" 
                      style={{ width: 16, height: 16 }}
                      checked={selectedToEnroll.includes(stu.id)}
                      onChange={(e) => {
                        if (e.target.checked) setSelectedToEnroll((prev) => [...prev, stu.id]);
                        else setSelectedToEnroll((prev) => prev.filter((id) => id !== stu.id));
                      }}
                    />
                    <div>
                      <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>{stu.name}</div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{stu.roll_no}</div>
                    </div>
                  </label>
                ))
              )}
            </div>

            <div className="modal-footer">
              <button className="btn btn-outline" onClick={() => setShowEnrollModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleEnroll} disabled={enrolling || availableStudents.length === 0}>
                {enrolling ? "Enrolling..." : `Enroll ${selectedToEnroll.length} Student(s)`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
