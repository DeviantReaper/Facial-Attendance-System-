import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import toast from "react-hot-toast";
import {
  Camera, Eye, EyeOff, Sun, Moon, ShieldCheck,
  BookOpen, Users, BarChart3, LogIn
} from "lucide-react";

const MOCK_USERS = [
  { username: "prof.sharma",  password: "professor123", role: "teacher", name: "Prof. Sharma" },
  { username: "prof.iyer",    password: "professor123", role: "teacher", name: "Prof. Iyer" },
  { username: "admin",        password: "admin123",     role: "admin",   name: "Admin" },
];

const MOCK_STUDENTS = [
  { roll_no: "MUJ2022001", password: "student123", name: "Arjun Mehta" },
  { roll_no: "MUJ2022002", password: "student123", name: "Priya Singh" },
  { roll_no: "MUJ2022003", password: "student123", name: "Rohit Kumar" },
];

export default function LoginPage() {
  const { login, mockLogin } = useAuth();
  const { theme, toggle } = useTheme();
  const navigate = useNavigate();

  const [tab, setTab] = useState("teacher");  // teacher | student
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showHints, setShowHints] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Try real backend first
      const userData = await login(username.trim(), password, tab === "student" ? "student" : "teacher");
      toast.success(`Welcome back, ${userData.name}!`);
    } catch (err) {
      // Fallback to mock if backend unreachable
      if (err?.code === "ERR_NETWORK" || err?.code === "ECONNREFUSED") {
        if (tab === "teacher") {
          const match = MOCK_USERS.find(
            (u) => u.username === username.trim().toLowerCase() && u.password === password
          );
          if (match) {
            mockLogin(match.role, { name: match.name });
            toast.success(`Welcome back, ${match.name}! (demo mode)`);
          } else {
            toast.error("Invalid credentials");
          }
        } else {
          const match = MOCK_STUDENTS.find(
            (s) => s.roll_no === username.trim().toUpperCase() && s.password === password
          );
          if (match) {
            mockLogin("student", { name: match.name, roll_no: match.roll_no });
            toast.success(`Welcome, ${match.name}! (demo mode)`);
          } else {
            toast.error("Invalid roll number or password");
          }
        }
      } else {
        const msg = err?.response?.data?.detail || "Invalid credentials";
        toast.error(msg);
      }
    }
    setLoading(false);
  };

  return (
    <div className="login-page" data-theme={theme === "dark" ? "dark" : undefined}>
      {/* Left branding panel */}
      <div className="login-left">
        {/* Theme toggle */}
        <button className="theme-toggle" onClick={toggle}
          style={{ position: "absolute", top: "1.5rem", right: "1.5rem" }}>
          {theme === "dark" ? <><Sun size={14}/> Light</> : <><Moon size={14}/> Dark</>}
        </button>

        <div className="login-branding animate-fade-in">
          <div className="logo-ring">
            <Camera size={40} color="#f97316" />
          </div>
          <h1>FaceAttend<span style={{ color: "var(--accent)" }}>.</span></h1>
          <p className="tagline">AI-Powered Attendance Management System</p>
          <p style={{ color: "rgba(255,255,255,0.5)", fontSize: "0.78rem", marginTop: "0.25rem" }}>
            Manipal University Jaipur · Academic Year 2025–26
          </p>
        </div>
      </div>

      {/* Right login panel */}
      <div className="login-right">
        <div className="login-form-panel animate-slide-up" style={{ width: "100%" }}>
          {/* Institution logo area */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "2rem" }}>
            <div style={{
              width: 44, height: 44, borderRadius: 10,
              background: "var(--primary)", display: "flex", alignItems: "center", justifyContent: "center"
            }}>
              <BookOpen size={20} color="#fff" />
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: "0.95rem", color: "var(--text)" }}>MUJ Portal</div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Attendance Management System</div>
            </div>
          </div>

          <h2>Welcome back</h2>
          <p className="subtitle">Sign in to continue to your portal</p>

          {/* Role tabs */}
          <div style={{
            display: "flex", background: "var(--bg)", borderRadius: "var(--radius)",
            padding: 4, marginBottom: "1.5rem",
          }}>
            {[["teacher", "Faculty / Admin"], ["student", "Student"]].map(([val, label]) => (
              <button key={val} onClick={() => setTab(val)} style={{
                flex: 1, padding: "0.55rem", border: "none", cursor: "pointer",
                borderRadius: "calc(var(--radius) - 2px)",
                fontFamily: "inherit", fontSize: "0.85rem", fontWeight: 600,
                background: tab === val ? "var(--bg-card)" : "transparent",
                color: tab === val ? "var(--primary)" : "var(--text-muted)",
                boxShadow: tab === val ? "var(--shadow-sm)" : "none",
                transition: "var(--transition)",
              }}>{label}</button>
            ))}
          </div>

          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div className="form-group">
              <label className="form-label">
                {tab === "teacher" ? "Username" : "Roll Number"}
              </label>
              <input
                className="form-input"
                type="text"
                placeholder={tab === "teacher" ? "e.g. prof.sharma" : "e.g. MUJ2022001"}
                value={username}
                autoComplete="username"
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <div style={{ position: "relative" }}>
                <input
                  className="form-input"
                  style={{ paddingRight: "2.75rem" }}
                  type={showPass ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  autoComplete="current-password"
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button type="button" onClick={() => setShowPass((v) => !v)} style={{
                  position: "absolute", right: "0.875rem", top: "50%", transform: "translateY(-50%)",
                  background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)",
                  display: "flex", padding: 0,
                }}>
                  {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button type="button" style={{
                background: "none", border: "none", color: "var(--primary)", fontSize: "0.8rem",
                cursor: "pointer", fontFamily: "inherit", fontWeight: 500,
              }}>Forgot password?</button>
            </div>

            <button type="submit" className="btn btn-primary w-full btn-lg" disabled={loading}>
              {loading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Signing in…</> : <><LogIn size={16} /> Sign In</>}
            </button>
          </form>

          {/* Demo hints */}
          <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
            <button type="button" onClick={() => setShowHints((v) => !v)} style={{
              background: "none", border: "none", color: "var(--text-muted)", fontSize: "0.78rem",
              cursor: "pointer", fontFamily: "inherit", textDecoration: "underline",
            }}>
              {showHints ? "Hide" : "Show"} demo credentials
            </button>
          </div>

          {showHints && (
            <div style={{
              marginTop: "0.75rem", background: "var(--bg)", border: "1px solid var(--border)",
              borderRadius: "var(--radius)", padding: "1rem", fontSize: "0.78rem",
              display: "flex", flexDirection: "column", gap: "0.5rem",
            }} className="animate-fade-in">
              <p style={{ fontWeight: 700, color: "var(--text-muted)" }}>👨‍🏫 Faculty — password: <code style={{ color: "var(--primary)" }}>professor123</code></p>
              <p style={{ fontFamily: "monospace" }}>prof.sharma &nbsp; prof.iyer &nbsp; admin (pw: admin123)</p>
              <div className="divider" />
              <p style={{ fontWeight: 700, color: "var(--text-muted)" }}>👨‍🎓 Student — password: <code style={{ color: "var(--accent)" }}>student123</code></p>
              <p style={{ fontFamily: "monospace" }}>MUJ2022001 &nbsp; MUJ2022002 &nbsp; MUJ2022003</p>
            </div>
          )}

          <p style={{ textAlign: "center", color: "var(--text-muted)", fontSize: "0.75rem", marginTop: "1.5rem" }}>
            New student?{" "}
            <a href="/enroll" style={{ color: "var(--primary)", fontWeight: 600, textDecoration: "none" }}>
              Self-enroll here
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
