import React, { useState, useRef, useEffect } from "react";
import { Camera, CheckCircle, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { studentsAPI, faceAPI } from "../services/api";

export default function RegisterStudent() {
  const navigate = useNavigate();
  
  const [name, setName] = useState("");
  const [rollNo, setRollNo] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    // Start camera immediately on mount
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = async () => {
             try {
                await videoRef.current.play();
             } catch(e) {
                console.error(e);
             }
          };
        }
      } catch (err) {
        console.error("Camera access error:", err);
      }
    };
    startCamera();

    return () => {
      // Cleanup camera on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const handleCapture = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0);
    setCapturedImage(canvas.toDataURL("image/png"));
  };

  const handleRetake = () => {
    setCapturedImage(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!capturedImage) {
      alert("Please capture a face reference before registering.");
      return;
    }

    try {
      // 1. Create Student record
      const sRes = await studentsAPI.create({
        name,
        roll_no: rollNo,
        email: `${rollNo.toLowerCase()}@muj.edu.in`
      });
      const studentId = sRes.data.id;

      // 2. Convert base64 capture to Blob
      const res = await fetch(capturedImage);
      const blob = await res.blob();
      const file = new File([blob], "face.png", { type: "image/png" });
      
      // 3. Register Face
      const fd = new FormData();
      fd.append("student_id", studentId);
      fd.append("files", file);
      
      await faceAPI.enroll(fd);

      // Stop camera
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      
      alert(`Successfully registered ${name}! You can now login or be marked present.`);
      navigate("/");
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.detail || "Failed to register. Roll number might already exist.");
    }
  };

  return (
    <div className="animate-slide-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <button
            onClick={() => navigate("/")}
            style={{ background: "var(--bg-card)", border: "1px solid var(--surface-border)", borderRadius: "8px", padding: "0.5rem", cursor: "pointer", color: "var(--text-muted)", display: "flex" }}
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <h2 style={{ fontSize: "1.8rem" }}>Add New Student</h2>
            <p style={{ color: "var(--text-muted)" }}>Register a student profile and capture their facial embedding.</p>
          </div>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
        {/* Form panel */}
        <form onSubmit={handleSubmit} className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          <h3 style={{ fontSize: "1.1rem" }}>Student Details</h3>
          
          <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Full Name</label>
            <input
              type="text"
              required
              placeholder="e.g. Aditi Sharma"
              className="search-input"
              value={name}
              onChange={v => setName(v.target.value)}
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Roll Number / ID</label>
            <input
              type="text"
              required
              placeholder="e.g. MUJ2022401"
              className="search-input"
              value={rollNo}
              onChange={v => setRollNo(v.target.value)}
              style={{ width: "100%" }}
            />
          </div>

          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "1rem" }}>
            Ensure the environment is well-lit and the student's face is clearly visible before capturing.
          </p>

          <button
            type="submit"
            className="btn-primary"
            style={{ marginTop: "auto", display: "flex", justifyContent: "center", gap: "0.5rem" }}
          >
            <CheckCircle size={18} /> Register Student
          </button>
        </form>

        {/* Camera panel */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
           <h3 style={{ fontSize: "1.1rem", alignSelf: "flex-start" }}>Face Registration</h3>
           
           <div style={{ position: "relative", width: "100%", aspectRatio: "4/3", background: "#000", borderRadius: "10px", overflow: "hidden" }}>
             {!capturedImage ? (
               <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
             ) : (
               <img src={capturedImage} alt="Captured face" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
             )}
           </div>
           
           {!capturedImage ? (
             <button type="button" className="btn-record" onClick={handleCapture} style={{ width: "100%", justifyContent: "center" }}>
               <Camera size={18} /> Capture Face
             </button>
           ) : (
             <button type="button" className="btn-secondary" onClick={handleRetake} style={{ width: "100%", justifyContent: "center", border: "1px solid var(--error)", color: "var(--error)" }}>
               Retake Capture
             </button>
           )}
        </div>
      </div>
    </div>
  );
}
