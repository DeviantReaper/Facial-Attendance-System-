import React, { useEffect, useRef, useState, useCallback } from "react";
import { Camera, CheckCircle, AlertCircle } from "lucide-react";

/**
 * AttendanceCamera — opens the webcam, draws bounding boxes,
 * and simulates facial recognition against a student roster.
 * 
 * Props:
 *   students  – array of { id, name, rollNo, avatar, status }
 *   onMark    – callback(studentId) called when a student is "recognized"
 *   isActive  – boolean, whether the camera session is running
 */
export default function AttendanceCamera({ students, onMark, isActive }) {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const [cameraReady, setCameraReady] = useState(false);
  const [recentScan, setRecentScan]   = useState(null); // { name, status }
  const [scanQueue, setScanQueue]     = useState([]);    // students not yet marked
  const [scanLog, setScanLog]         = useState([]);    // all scans so far

  // Build queue of absent students when session starts
  useEffect(() => {
    if (isActive) {
      const absent = students.filter(s => s.status === "absent");
      setScanQueue(absent);
      setScanLog([]);
      setRecentScan(null);
    } else {
      stopCamera();
    }
  }, [isActive]);

  // Start camera when active
  useEffect(() => {
    if (isActive) startCamera();
    return () => stopCamera();
  }, [isActive]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = async () => {
          try {
            await videoRef.current.play();
            setCameraReady(true);
            // Start simulated recognition loop after camera plays
            startRecognitionLoop();
          } catch (e) {
            console.error("Video play failed:", e);
          }
        };
      }
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setCameraReady(false);
  };

  const startRecognitionLoop = () => {
    // Every 2.5 seconds, "recognize" the next student in the queue
    let idx = 0;
    intervalRef.current = setInterval(() => {
      setScanQueue(prev => {
        if (prev.length === 0) {
          clearInterval(intervalRef.current);
          return prev;
        }
        const student = prev[idx % prev.length];
        if (!student) return prev;

        // "Recognize" this student
        setRecentScan({ name: student.name, status: "recognized" });
        setScanLog(log => [{ name: student.name, time: new Date().toLocaleTimeString(), status: "present" }, ...log]);
        onMark(student.id);

        // Remove from queue
        const next = prev.filter(s => s.id !== student.id);
        if (next.length === 0) {
          clearInterval(intervalRef.current);
        }
        return next;
      });
    }, 2500);
  };

  // Draw bounding box overlay on canvas
  useEffect(() => {
    if (!cameraReady || !isActive) return;

    const drawLoop = setInterval(() => {
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      const ctx = canvas.getContext("2d");
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw a detection box in the center (simulating face detection)
      const cx = canvas.width  / 2;
      const cy = canvas.height / 2;
      const bw = 160, bh = 200;

      const color = recentScan ? "#22c55e" : "#6366f1";
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(cx - bw/2, cy - bh/2, bw, bh);

      // Draw scan indicator corners
      const cornerLen = 20;
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      // top-left
      ctx.beginPath(); ctx.moveTo(cx-bw/2, cy-bh/2+cornerLen); ctx.lineTo(cx-bw/2, cy-bh/2); ctx.lineTo(cx-bw/2+cornerLen, cy-bh/2); ctx.stroke();
      // top-right
      ctx.beginPath(); ctx.moveTo(cx+bw/2-cornerLen, cy-bh/2); ctx.lineTo(cx+bw/2, cy-bh/2); ctx.lineTo(cx+bw/2, cy-bh/2+cornerLen); ctx.stroke();
      // bottom-left
      ctx.beginPath(); ctx.moveTo(cx-bw/2, cy+bh/2-cornerLen); ctx.lineTo(cx-bw/2, cy+bh/2); ctx.lineTo(cx-bw/2+cornerLen, cy+bh/2); ctx.stroke();
      // bottom-right
      ctx.beginPath(); ctx.moveTo(cx+bw/2-cornerLen, cy+bh/2); ctx.lineTo(cx+bw/2, cy+bh/2); ctx.lineTo(cx+bw/2, cy+bh/2-cornerLen); ctx.stroke();

      // Label
      if (recentScan) {
        ctx.fillStyle = "rgba(34,197,94,0.85)";
        ctx.fillRect(cx - bw/2, cy - bh/2 - 32, bw, 30);
        ctx.fillStyle = "white";
        ctx.font = "bold 14px Outfit, sans-serif";
        ctx.fillText(recentScan.name, cx - bw/2 + 6, cy - bh/2 - 10);
      } else {
        ctx.fillStyle = "rgba(99,102,241,0.75)";
        ctx.fillRect(cx - bw/2, cy - bh/2 - 32, bw, 30);
        ctx.fillStyle = "white";
        ctx.font = "bold 13px Outfit, sans-serif";
        ctx.fillText("Scanning…", cx - bw/2 + 6, cy - bh/2 - 10);
      }

    }, 50); // 20fps render

    return () => clearInterval(drawLoop);
  }, [cameraReady, isActive, recentScan]);

  if (!isActive) return null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: "1rem", marginTop: "0.5rem" }}>
      {/* Camera feed */}
      <div className="glass-panel" style={{ padding: "0.75rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "1rem" }}>
            <Camera size={18} /> Live Recognition
          </h3>
          <span className="status-badge status-success">● Camera Active</span>
        </div>
        <div style={{ position: "relative", borderRadius: "10px", overflow: "hidden", background: "#000" }}>
          <video ref={videoRef} autoPlay playsInline muted style={{ display: "none" }} />
          <canvas ref={canvasRef} style={{ width: "100%", height: "auto", display: "block" }} />
        </div>
        <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", textAlign: "center" }}>
          Align your face within the frame. Students are recognized automatically.
        </p>
      </div>

      {/* Scan Log */}
      <div className="glass-panel" style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem", maxHeight: "500px", overflow: "auto" }}>
        <h3 style={{ fontSize: "0.95rem" }}>Recognition Log</h3>

        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "var(--text-muted)" }}>
          <span>Remaining: {scanQueue.length}</span>
          <span>Marked: {scanLog.length}</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          {scanLog.length === 0 && (
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>Waiting for faces…</p>
          )}
          {scanLog.map((s, i) => (
            <div key={i} style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "0.5rem 0.65rem", background: "rgba(34,197,94,0.08)", borderRadius: "8px",
              fontSize: "0.82rem", animation: i === 0 ? "fadeIn 0.3s ease" : undefined,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <CheckCircle size={14} color="var(--success)" />
                <span style={{ fontWeight: 500 }}>{s.name}</span>
              </div>
              <span style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>{s.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
