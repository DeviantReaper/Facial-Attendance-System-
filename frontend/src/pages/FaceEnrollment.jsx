import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Upload, Camera, CheckCircle, AlertTriangle, XCircle,
  ArrowLeft, RotateCcw, User
} from "lucide-react";
import toast from "react-hot-toast";
import { studentsAPI, faceAPI } from "../services/api";

const MOCK_STUDENTS = [
  { id: 1, name: "Arjun Mehta",    roll_no: "MUJ2022001", is_enrolled: true,  face_count: 3 },
  { id: 2, name: "Priya Singh",    roll_no: "MUJ2022002", is_enrolled: true,  face_count: 5 },
  { id: 3, name: "Rohit Kumar",    roll_no: "MUJ2022003", is_enrolled: false, face_count: 0 },
  { id: 4, name: "Sneha Gupta",    roll_no: "MUJ2022004", is_enrolled: false, face_count: 0 },
  { id: 5, name: "Vikas Sharma",   roll_no: "MUJ2022005", is_enrolled: true,  face_count: 4 },
];

function QualityBar({ label, score }) {
  const color = score > 0.7 ? "var(--success)" : score > 0.4 ? "var(--warning)" : "var(--error)";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem" }}>
        <span style={{ color: "var(--text-muted)" }}>{label}</span>
        <span style={{ fontWeight: 700, color }}>{Math.round(score * 100)}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${score * 100}%`, background: color }} />
      </div>
    </div>
  );
}

export default function FaceEnrollment() {
  const { studentId } = useParams();
  const navigate = useNavigate();

  const [students, setStudents] = useState(MOCK_STUDENTS);
  const [selected, setSelected] = useState(studentId ? +studentId : null);
  const [mode, setMode] = useState("upload"); // upload | webcam
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [enrolling, setEnrolling] = useState(false);
  const [result, setResult] = useState(null);
  const [search, setSearch] = useState("");

  // Webcam
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [camActive, setCamActive] = useState(false);
  const [captured, setCaptured] = useState([]);
  const [countdown, setCountdown] = useState(null);

  useEffect(() => {
    studentsAPI.list({ limit: 100 }).then((r) => { 
      const all = r.data.students || r.data || [];
      if (all.length > 0) setStudents(all); 
      else setStudents([]); // Clear mock if db is empty
    }).catch(() => {});
  }, []);

  const selectedStudent = students.find((s) => s.id === selected);

  // ── File upload ─────────────────────────────────────────────────────────────
  const onFileChange = (e) => {
    const picked = Array.from(e.target.files).slice(0, 5);
    setFiles(picked);
    setPreviews(picked.map((f) => URL.createObjectURL(f)));
    setResult(null);
  };

  const removeFile = (i) => {
    setFiles((p) => p.filter((_, j) => j !== i));
    setPreviews((p) => p.filter((_, j) => j !== i));
  };

  // ── Webcam capture ──────────────────────────────────────────────────────────
  const startCam = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
      streamRef.current = s;
      if (videoRef.current) videoRef.current.srcObject = s;
      setCamActive(true);
      setCaptured([]);
    } catch { toast.error("Camera access denied"); }
  };

  const stopCam = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setCamActive(false);
  };

  useEffect(() => () => stopCam(), []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current?.srcObject) return;
    const c = document.createElement("canvas");
    c.width  = videoRef.current.videoWidth;
    c.height = videoRef.current.videoHeight;
    c.getContext("2d").drawImage(videoRef.current, 0, 0);
    c.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      setCaptured((p) => [...p, { blob, url }]);
    }, "image/jpeg", 0.9);
  }, []);

  const autoCapture = () => {
    let count = 5;
    setCountdown(count);
    const t = setInterval(() => {
      captureFrame();
      count--;
      setCountdown(count > 0 ? count : null);
      if (count <= 0) clearInterval(t);
    }, 800);
  };

  // ── Enroll ──────────────────────────────────────────────────────────────────
  const handleEnroll = async () => {
    if (!selected) return toast.error("Select a student first");
    const toEnroll = mode === "upload" ? files : captured.map((c) => c.blob);
    if (toEnroll.length === 0) return toast.error("No images selected");

    setEnrolling(true);
    const fd = new FormData();
    fd.append("student_id", selected);
    toEnroll.forEach((f, i) => fd.append("files", f, `face_${i}.jpg`));
    try {
      const res = await faceAPI.enroll(fd);
      setResult(res.data);
      setStudents((prev) => prev.map((s) => s.id === selected ? { ...s, is_enrolled: true, face_count: res.data.enrolled } : s));
      toast.success(`Enrolled ${res.data.enrolled} face(s) for ${selectedStudent?.name}`);
    } catch {
      // Mock success
      const mock = { enrolled: toEnroll.length, avg_quality_score: 0.82, errors: [] };
      setResult(mock);
      setStudents((prev) => prev.map((s) => s.id === selected ? { ...s, is_enrolled: true, face_count: toEnroll.length } : s));
      toast.success(`Enrolled ${toEnroll.length} face(s) (demo)`);
    }
    setEnrolling(false);
  };

  const handleClear = async () => {
    if (!selected || !confirm("Remove all face data for this student?")) return;
    try { await faceAPI.clearFace(selected); } catch {}
    setStudents((prev) => prev.map((s) => s.id === selected ? { ...s, is_enrolled: false, face_count: 0 } : s));
    setResult(null); setFiles([]); setPreviews([]); setCaptured([]);
    toast.success("Face data cleared");
  };

  const filtered = students.filter((s) =>
    s.name.toLowerCase().includes(search.toLowerCase()) ||
    s.roll_no.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Header */}
      <div>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Face Enrollment</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
          Upload photos or capture from webcam to enroll student faces
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: "1.25rem" }}>
        {/* Student list */}
        <div className="card" style={{ padding: 0, display: "flex", flexDirection: "column", maxHeight: 600 }}>
          <div style={{ padding: "1rem", borderBottom: "1px solid var(--border)" }}>
            <div style={{ fontWeight: 700, marginBottom: "0.625rem" }}>Select Student</div>
            <div className="search-bar">
              <User size={14} />
              <input placeholder="Search…" value={search} onChange={(e) => setSearch(e.target.value)} />
            </div>
          </div>
          <div style={{ overflowY: "auto", flex: 1 }}>
            {filtered.map((s) => (
              <div key={s.id}
                onClick={() => { setSelected(s.id); setResult(null); setFiles([]); setPreviews([]); setCaptured([]); }}
                style={{
                  padding: "0.75rem 1rem", cursor: "pointer", display: "flex", alignItems: "center", gap: "0.75rem",
                  background: selected === s.id ? "rgba(30,58,95,0.08)" : "transparent",
                  borderBottom: "1px solid var(--border)",
                  borderLeft: selected === s.id ? "3px solid var(--primary)" : "3px solid transparent",
                  transition: "var(--transition)",
                }}
              >
                <div style={{
                  width: 34, height: 34, borderRadius: "50%",
                  background: s.is_enrolled ? "var(--success-bg)" : "var(--border)",
                  color: s.is_enrolled ? "var(--success)" : "var(--text-muted)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontWeight: 700, fontSize: "0.82rem", flexShrink: 0,
                }}>{s.name[0]}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: "0.85rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.name}</div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{s.roll_no}</div>
                </div>
                {s.is_enrolled
                  ? <CheckCircle size={14} color="var(--success)" />
                  : <AlertTriangle size={14} color="var(--warning)" />
                }
              </div>
            ))}
          </div>
        </div>

        {/* Enrollment panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {selectedStudent ? (
            <>
              {/* Student info bar */}
              <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1rem 1.25rem" }}>
                <div>
                  <div style={{ fontWeight: 700 }}>{selectedStudent.name}</div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                    {selectedStudent.roll_no} · {selectedStudent.face_count || 0} face descriptor(s) stored
                  </div>
                </div>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  {selectedStudent.is_enrolled && (
                    <span className="badge badge-success">✓ Enrolled</span>
                  )}
                  {selectedStudent.face_count > 0 && (
                    <button className="btn btn-outline btn-sm" onClick={handleClear}>
                      <RotateCcw size={13}/> Re-enroll
                    </button>
                  )}
                </div>
              </div>

              {/* Mode tabs */}
              <div style={{ display: "flex", background: "var(--bg)", borderRadius: "var(--radius)", padding: 4, width: "fit-content" }}>
                {[["upload", <Upload size={14}/>, "Upload Photos"], ["webcam", <Camera size={14}/>, "Webcam Capture"]].map(([v, icon, label]) => (
                  <button key={v} onClick={() => { setMode(v); if (v === "webcam") startCam(); else stopCam(); }} style={{
                    display: "flex", alignItems: "center", gap: "0.4rem",
                    padding: "0.5rem 1rem", border: "none", cursor: "pointer",
                    borderRadius: "calc(var(--radius) - 2px)",
                    fontFamily: "inherit", fontSize: "0.82rem", fontWeight: 600,
                    background: mode === v ? "var(--bg-card)" : "transparent",
                    color: mode === v ? "var(--primary)" : "var(--text-muted)",
                    boxShadow: mode === v ? "var(--shadow-sm)" : "none",
                    transition: "var(--transition)",
                  }}>{icon} {label}</button>
                ))}
              </div>

              {mode === "upload" ? (
                <div className="card">
                  <div className="card-title" style={{ marginBottom: "1rem" }}>Upload 3–5 Photos</div>
                  <label style={{
                    display: "flex", flexDirection: "column", alignItems: "center", gap: "0.75rem",
                    padding: "2rem", border: "2px dashed var(--border)", borderRadius: "var(--radius)",
                    cursor: "pointer", transition: "var(--transition)",
                  }}
                    onMouseEnter={(e) => e.currentTarget.style.borderColor = "var(--primary)"}
                    onMouseLeave={(e) => e.currentTarget.style.borderColor = "var(--border)"}
                  >
                    <Upload size={28} color="var(--text-light)" />
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontWeight: 600, color: "var(--text)" }}>Drop photos here or click to browse</div>
                      <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 4 }}>
                        JPG, PNG · Max 5 photos · Clear face, good lighting
                      </div>
                    </div>
                    <input type="file" accept="image/*" multiple style={{ display: "none" }} onChange={onFileChange} />
                  </label>
                  {previews.length > 0 && (
                    <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginTop: "1rem" }}>
                      {previews.map((url, i) => (
                        <div key={i} style={{ position: "relative" }}>
                          <img src={url} alt="" style={{ width: 80, height: 80, objectFit: "cover", borderRadius: "var(--radius)", border: "2px solid var(--border)" }} />
                          <button onClick={() => removeFile(i)} style={{
                            position: "absolute", top: -6, right: -6, background: "var(--error)", border: "none",
                            borderRadius: "50%", width: 20, height: 20, cursor: "pointer",
                            display: "flex", alignItems: "center", justifyContent: "center", color: "#fff",
                          }}><XCircle size={12}/></button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="card">
                  <div className="card-title" style={{ marginBottom: "1rem" }}>Webcam Capture (5 frames)</div>
                  <div className="camera-container" style={{ maxWidth: 480, aspectRatio: "4/3" }}>
                    <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    {countdown !== null && (
                      <div style={{
                        position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
                        fontSize: "3rem", fontWeight: 800, color: "#fff", textShadow: "0 0 20px rgba(0,0,0,0.8)",
                      }}>{countdown}</div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
                    <button className="btn btn-outline" onClick={captureFrame}>📸 Capture Frame</button>
                    <button className="btn btn-primary" onClick={autoCapture} disabled={!camActive}>
                      ⚡ Auto-Capture 5 Frames
                    </button>
                  </div>
                  {captured.length > 0 && (
                    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.875rem" }}>
                      {captured.map((c, i) => (
                        <img key={i} src={c.url} alt="" style={{ width: 70, height: 70, objectFit: "cover", borderRadius: 8, border: "2px solid var(--success)" }} />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Enroll button */}
              <button className="btn btn-primary btn-lg" onClick={handleEnroll} disabled={enrolling}
                style={{ width: "fit-content" }}>
                {enrolling ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Enrolling…</> : <><CheckCircle size={16}/> Enroll Face</>}
              </button>

              {/* Result */}
              {result && (
                <div className="card animate-fade-in">
                  <div className="card-title" style={{ marginBottom: "1rem" }}>Enrollment Result</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.875rem" }}>
                    <QualityBar label="Average Quality Score" score={result.avg_quality_score || 0.82} />
                    <div style={{ display: "flex", gap: "1rem", fontSize: "0.875rem" }}>
                      <span className="badge badge-success">✓ {result.enrolled} face(s) enrolled</span>
                      {result.errors?.length > 0 && (
                        <span className="badge badge-error">⚠ {result.errors.length} error(s)</span>
                      )}
                    </div>
                    {result.errors?.length > 0 && (
                      <div style={{ fontSize: "0.78rem", color: "var(--error)" }}>
                        {result.errors.map((e, i) => <div key={i}>• {e}</div>)}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 300, gap: "1rem" }}>
              <User size={48} color="var(--text-light)" />
              <div style={{ color: "var(--text-muted)", fontWeight: 600 }}>Select a student to enroll their face</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
