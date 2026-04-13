import React, { useState } from "react";
import { Sun, Moon, Eye, EyeOff, LogIn } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import { mockStudents } from "../data/mockData";

// Helper to compute dynamic mock users list
const getMockUsers = () => [
  { username: "prof.sharma",  password: "professor123", role: "professor", user: { name: "Prof. Sharma"    } },
  { username: "prof.iyer",    password: "professor123", role: "professor", user: { name: "Prof. Iyer"      } },
  ...mockStudents.map(s => ({
    username: s.rollNo.toLowerCase(),
    password: "student123",
    role: "student",
    user: s,
  })),
];

export default function RoleSelect() {
  const { login }  = useAuth();
  const { theme, toggle } = useTheme();

  const [username, setUsername]   = useState("");
  const [password, setPassword]   = useState("");
  const [showPass, setShowPass]   = useState(false);
  const [error, setError]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [showHints, setShowHints] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    setTimeout(() => {
      const match = getMockUsers().find(
        u => u.username === username.trim().toLowerCase() && u.password === password
      );
      if (match) {
        login(match.role, match.user);
      } else {
        setError("Invalid username or password. Try the hints below.");
      }
      setLoading(false);
    }, 600);
  };

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: "1.5rem",
      padding: "2rem",
      background: "var(--bg-dark)",
      backgroundImage: "radial-gradient(circle at 20% 50%, var(--body-gradient-1) 0%, transparent 30%), radial-gradient(circle at 80% 20%, var(--body-gradient-2) 0%, transparent 30%)",
      position: "relative",
    }}>

      {/* Theme toggle */}
      <button className="theme-toggle" onClick={toggle} style={{ position: "absolute", top: "1.5rem", right: "1.5rem" }}>
        {theme === "dark" ? <><Sun size={14}/> Light</> : <><Moon size={14}/> Dark</>}
      </button>

      {/* Brand */}
      <div style={{ textAlign: "center" }}>
        <h1 className="text-gradient" style={{ fontSize: "2.8rem", letterSpacing: "-0.04em" }}>NeoAttend</h1>
      </div>

      {/* Login card */}
      <form
        onSubmit={handleSubmit}
        className="glass-panel animate-fade-in"
        style={{ width: "100%", maxWidth: "380px", display: "flex", flexDirection: "column", gap: "1.1rem" }}
      >
        <div>
          <h2 style={{ fontSize: "1.3rem", marginBottom: "0.25rem" }}>Welcome back</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Sign in to your account</p>
        </div>

        {/* Username */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          <label style={{ fontSize: "0.82rem", color: "var(--text-muted)", fontWeight: 500 }}>Username</label>
          <input
            className="search-input"
            style={{ width: "100%" }}
            type="text"
            placeholder="e.g. prof.sharma or muj2022101"
            value={username}
            autoComplete="username"
            onChange={e => setUsername(e.target.value)}
            required
          />
        </div>

        {/* Password */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          <label style={{ fontSize: "0.82rem", color: "var(--text-muted)", fontWeight: 500 }}>Password</label>
          <div style={{ position: "relative" }}>
            <input
              className="search-input"
              style={{ width: "100%", paddingRight: "2.5rem" }}
              type={showPass ? "text" : "password"}
              placeholder="Enter your password"
              value={password}
              autoComplete="current-password"
              onChange={e => setPassword(e.target.value)}
              required
            />
            <button
              type="button"
              onClick={() => setShowPass(v => !v)}
              style={{ position: "absolute", right: "0.75rem", top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", display: "flex" }}
            >
              {showPass ? <EyeOff size={16}/> : <Eye size={16}/>}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <p style={{ color: "var(--error)", fontSize: "0.82rem", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)", borderRadius: "8px", padding: "0.5rem 0.75rem" }}>
            {error}
          </p>
        )}

        {/* Submit */}
        <button
          type="submit"
          className="btn-primary"
          disabled={loading}
          style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem", opacity: loading ? 0.7 : 1 }}
        >
          <LogIn size={16}/>
          {loading ? "Signing in…" : "Sign In"}
        </button>

        {/* Hints toggle */}
        <button
          type="button"
          onClick={() => setShowHints(v => !v)}
          style={{ background: "none", border: "none", color: "var(--text-muted)", fontSize: "0.8rem", cursor: "pointer", textDecoration: "underline" }}
        >
          {showHints ? "Hide" : "Show"} mock credentials
        </button>

        {showHints && (
          <div style={{ background: "var(--bg-card)", border: "1px solid var(--surface-border)", borderRadius: "10px", padding: "0.9rem", fontSize: "0.78rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <p style={{ fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.25rem" }}>👨‍🏫 Professor accounts — password: <code style={{ color: "var(--primary)" }}>professor123</code></p>
            <p style={{ fontFamily: "monospace", color: "var(--text-main)" }}>prof.sharma</p>
            <p style={{ fontFamily: "monospace", color: "var(--text-main)" }}>prof.iyer</p>
            <hr style={{ border: "none", borderTop: "1px solid var(--surface-border)", margin: "0.25rem 0" }}/>
            <p style={{ fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.25rem" }}>👨‍🎓 Student accounts — password: <code style={{ color: "#ec4899" }}>student123</code></p>
            {mockStudents.map(s => (
              <p key={s.id} style={{ fontFamily: "monospace", color: "var(--text-main)" }}>
                {s.rollNo.toLowerCase()} — <span style={{ color: "var(--text-muted)" }}>{s.name}</span>
              </p>
            ))}
          </div>
        )}
      </form>

      <p style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
        Mock prototype — no real authentication
      </p>
      <button 
        onClick={() => {
          // Explicitly stop any leftover streams before navigating
          window.location.href = "/register";
        }}
        style={{ background: "none", border: "none", color: "var(--primary)", fontSize: "0.85rem", cursor: "pointer", textDecoration: "underline", fontWeight: 500 }}
      >
        New Student? Register Face
      </button>
    </div>
  );
}

