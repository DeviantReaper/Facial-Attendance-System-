import React, { useState } from "react";
import { Save, Building2, Shield, Bell, Palette } from "lucide-react";
import toast from "react-hot-toast";
import { useTheme } from "../context/ThemeContext";

export default function Settings() {
  const { theme, toggle } = useTheme();
  const [institution, setInstitution] = useState({
    name: "Manipal University Jaipur",
    short_name: "MUJ",
    academic_year: "2025-26",
    address: "Dehmi Kalan, Near GVK Toll Plaza, Jaipur, Rajasthan 303007",
  });
  const [rules, setRules] = useState({
    min_attendance_pct: 75,
    late_threshold_minutes: 10,
    cooldown_minutes: 5,
    recognition_threshold: 0.75,
  });
  const [email, setEmail] = useState({
    enabled: false,
    smtp_host: "smtp.gmail.com",
    smtp_port: 587,
    smtp_user: "",
    from_email: "attendance@muj.edu",
  });

  const save = (section) => toast.success(`${section} settings saved`);

  const IR = (k) => (e) => setInstitution((p) => ({ ...p, [k]: e.target.value }));
  const RR = (k) => (e) => setRules((p) => ({ ...p, [k]: e.target.type === "number" ? +e.target.value : e.target.value }));
  const ER = (k) => (e) => setEmail((p) => ({ ...p, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));

  const Tab = ({ label, icon }) => (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontWeight: 800, fontSize: "1rem" }}>
        {icon} {label}
      </div>
    </div>
  );

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Settings</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
          Configure institution, attendance rules, and notification settings
        </p>
      </div>

      {/* Institution */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Building2 size={18} color="var(--primary)" />
            <div className="card-title">Institution Settings</div>
          </div>
          <button className="btn btn-primary btn-sm" onClick={() => save("Institution")}><Save size={14}/> Save</button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Institution Name</label>
              <input className="form-input" value={institution.name} onChange={IR("name")} />
            </div>
            <div className="form-group">
              <label className="form-label">Short Name / Code</label>
              <input className="form-input" value={institution.short_name} onChange={IR("short_name")} />
            </div>
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Academic Year</label>
              <input className="form-input" value={institution.academic_year} onChange={IR("academic_year")} placeholder="2025-26" />
            </div>
          </div>
          <div className="form-group">
            <label className="form-label">Address</label>
            <textarea className="form-textarea" style={{ minHeight: 70 }} value={institution.address} onChange={IR("address")} />
          </div>
        </div>
      </div>

      {/* Attendance Rules */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Shield size={18} color="var(--accent)" />
            <div className="card-title">Attendance Rules</div>
          </div>
          <button className="btn btn-primary btn-sm" onClick={() => save("Attendance rules")}><Save size={14}/> Save</button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Minimum Attendance %</label>
              <input className="form-input" type="number" min={0} max={100} value={rules.min_attendance_pct} onChange={RR("min_attendance_pct")} />
              <span className="form-hint">Students below this are flagged as defaulters</span>
            </div>
            <div className="form-group">
              <label className="form-label">Late Threshold (minutes)</label>
              <input className="form-input" type="number" min={0} max={60} value={rules.late_threshold_minutes} onChange={RR("late_threshold_minutes")} />
              <span className="form-hint">Arrivals after this many minutes are marked "Late"</span>
            </div>
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Attendance Cooldown (minutes)</label>
              <input className="form-input" type="number" min={1} max={60} value={rules.cooldown_minutes} onChange={RR("cooldown_minutes")} />
              <span className="form-hint">Prevents duplicate entries within this window</span>
            </div>
            <div className="form-group">
              <label className="form-label">Recognition Confidence Threshold</label>
              <input className="form-input" type="number" step={0.05} min={0.5} max={1} value={rules.recognition_threshold} onChange={RR("recognition_threshold")} />
              <span className="form-hint">Min confidence (0–1) to auto-mark present</span>
            </div>
          </div>
        </div>
      </div>

      {/* Email Notifications */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Bell size={18} color="var(--success)" />
            <div className="card-title">Email Notifications</div>
          </div>
          <button className="btn btn-primary btn-sm" onClick={() => save("Email")}><Save size={14}/> Save</button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "0.875rem", cursor: "pointer" }}>
            <input type="checkbox" checked={email.enabled} onChange={ER("enabled")} style={{ accentColor: "var(--primary)", width: 16, height: 16 }} />
            <div>
              <div style={{ fontWeight: 600, fontSize: "0.875rem" }}>Enable Email Notifications</div>
              <div className="form-hint">Send absence alerts and enrollment confirmations via SMTP</div>
            </div>
          </label>
          {email.enabled && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }} className="animate-fade-in">
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">SMTP Host</label>
                  <input className="form-input" value={email.smtp_host} onChange={ER("smtp_host")} />
                </div>
                <div className="form-group">
                  <label className="form-label">SMTP Port</label>
                  <input className="form-input" type="number" value={email.smtp_port} onChange={ER("smtp_port")} />
                </div>
              </div>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">SMTP Username</label>
                  <input className="form-input" type="email" value={email.smtp_user} onChange={ER("smtp_user")} placeholder="your@gmail.com" />
                </div>
                <div className="form-group">
                  <label className="form-label">From Email</label>
                  <input className="form-input" type="email" value={email.from_email} onChange={ER("from_email")} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Appearance */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Palette size={18} color="var(--info)" />
            <div className="card-title">Appearance</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: "0.875rem" }}>Dark Mode</div>
            <div className="form-hint">Switch between light and dark interface</div>
          </div>
          <button className="btn btn-outline" onClick={toggle}>
            {theme === "dark" ? "☀️ Switch to Light" : "🌙 Switch to Dark"}
          </button>
        </div>
      </div>
    </div>
  );
}
