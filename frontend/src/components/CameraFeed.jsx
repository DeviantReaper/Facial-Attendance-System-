import React, { useEffect, useRef, useState } from "react";
import { Camera, AlertCircle, CheckCircle } from "lucide-react";

export default function CameraFeed({ classId }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [detections, setDetections] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket
    wsRef.current = new WebSocket(`ws://localhost:8000/ws/recognize`);
    
    wsRef.current.onopen = () => setIsConnected(true);
    wsRef.current.onclose = () => setIsConnected(false);

    wsRef.current.onmessage = (e) => {
      const { faces } = JSON.parse(e.data);
      if (faces) {
        setDetections(faces);
        drawBoundingBoxes(faces);
      }
    };

    // Initialize Camera
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        startStreaming();
      })
      .catch((err) => console.error("Camera access denied:", err));

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startStreaming = () => {
    setInterval(() => {
      if (!canvasRef.current || !videoRef.current || wsRef.current?.readyState !== WebSocket.OPEN) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      
      // Ensure canvas matches video dimensions
      if (canvas.width !== videoRef.current.videoWidth && videoRef.current.videoWidth > 0) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
      }
      
      // Draw current video frame to canvas
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      
      // Send frame to server
      canvas.toBlob((blob) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then((buf) => wsRef.current.send(buf));
        }
      }, "image/jpeg", 0.8);
      
    }, 200);  // 5 fps
  };

  const drawBoundingBoxes = (faces) => {
    if (!canvasRef.current || !videoRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Re-draw video frame
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    faces.forEach(face => {
      const [x1, y1, x2, y2] = face.bbox;
      const width = x2 - x1;
      const height = y2 - y1;
      
      // Choose color based on status
      let color = "#ef4444"; // red for spoof/unknown
      if (face.status === "marked") color = "#22c55e"; // green
      else if (face.status === "new_live_face") color = "#3b82f6"; // blue
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, width, height);
      
      // Draw Label
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 30, width, 30);
      ctx.fillStyle = "white";
      ctx.font = "16px Arial";
      ctx.fillText(
        face.name || face.status,
        x1 + 5,
        y1 - 10
      );
    });
  };

  return (
    <div className="glass-panel animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Camera size={24} className="text-gradient" /> Live Recognition
        </h2>
        <span className={`status-badge ${isConnected ? 'status-success' : 'status-error'}`}>
          {isConnected ? "Live stream active" : "Connecting..."}
        </span>
      </div>
      
      <div style={{ position: 'relative', width: '100%', borderRadius: '12px', overflow: 'hidden', background: '#000' }}>
        {/* Hidden video element for source */}
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          style={{ display: 'none' }} 
        />
        {/* Canvas used for rendering and extraction */}
        <canvas 
          ref={canvasRef} 
          style={{ width: '100%', height: 'auto', display: 'block' }}
        />
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
        <div style={{ flex: 1 }} className="glass-panel">
          <h3 style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>Recent Scans</h3>
          <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {detections.length === 0 && <p style={{ fontSize: '0.875rem' }}>Awaiting subjects...</p>}
            {detections.map((d, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '6px' }}>
                <span style={{ fontWeight: 500 }}>{d.name || 'Unknown Subject'}</span>
                {d.status === "marked" ? 
                  <CheckCircle size={18} color="var(--success)" /> : 
                  <AlertCircle size={18} color="var(--error)" />
                }
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
