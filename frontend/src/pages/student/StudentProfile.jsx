import React, { useState } from "react";
import { useAuth } from "../../context/AuthContext";
import { User, Mail, Phone, BookOpen, Key, Save } from "lucide-react";
import toast from "react-hot-toast";

export default function StudentProfile() {
  const { auth } = useAuth();
  const [form, setForm] = useState({
    name: auth?.name || "Arjun Mehta",
    email: "arjun@muj.edu",
    phone: "+91 9872345678",
    department: "CSE",
    semester: "6",
  });
  const [pwForm, setPwForm] = useState({ current: "", new: "", confirm: "" });
  const [saving, setSaving] = useState(false);

  const F = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const PF = (k) => (e) => setPwForm((f) => ({ ...f, [k]: e.target.value }));

  const handleSave = () => { setSaving(true); setTimeout(() => { toast.success("Profile updated"); setSaving(false); }, 800); };

  const handlePw = () => {
    if (!pwForm.current) return toast.error("Enter current password");
    if (pwForm.new !== pwForm.confirm) return toast.error("Passwords do not match");
    if (pwForm.new.length < 8) return toast.error("Password must be at least 8 characters");
    toast.success("Password changed successfully");
    setPwForm({ current: "", new: "", confirm: "" });
  };

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>My Profile</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
          Update your personal information and password
        </p>
      </div>

      {/* Avatar + info */}
      <div className="card" style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
        <div style={{
          width: 80, height: 80, borderRadius: "50%",
          background: "var(--primary)", color: "#fff",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontWeight: 800, fontSize: "2rem", flexShrink: 0,
        }}>{form.name[0]}</div>
        <div>
          <div style={{ fontWeight: 800, fontSize: "1.1rem" }}>{form.name}</div>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>
            {form.department} · Semester {form.semester}
          </div>
          <span className="badge badge-success" style={{ marginTop: "0.5rem", display: "inline-flex" }}>✓ Face Enrolled</span>
        </div>
      </div>

      {/* Edit form */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <User size={18} color="var(--primary)" />
            <div className="card-title">Personal Details</div>
          </div>
          <button className="btn btn-primary btn-sm" onClick={handleSave} disabled={saving}>
            {saving ? <span className="spinner" style={{ width: 13, height: 13 }} /> : <Save size={14}/>} Save
          </button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <input className="form-input" value={form.name} onChange={F("name")} />
            </div>
            <div className="form-group">
              <label className="form-label">Email</label>
              <input className="form-input" type="email" value={form.email} onChange={F("email")} />
            </div>
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Phone</label>
              <input className="form-input" value={form.phone} onChange={F("phone")} />
            </div>
            <div className="form-group">
              <label className="form-label">Department</label>
              <input className="form-input" value={form.department} readOnly style={{ opacity: 0.6 }} />
            </div>
          </div>
        </div>
      </div>

      {/* Change password */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Key size={18} color="var(--accent)" />
            <div className="card-title">Change Password</div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="form-group">
            <label className="form-label">Current Password</label>
            <input className="form-input" type="password" value={pwForm.current} onChange={PF("current")} placeholder="••••••••" />
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">New Password</label>
              <input className="form-input" type="password" value={pwForm.new} onChange={PF("new")} placeholder="Min 8 characters" />
            </div>
            <div className="form-group">
              <label className="form-label">Confirm New Password</label>
              <input className="form-input" type="password" value={pwForm.confirm} onChange={PF("confirm")} placeholder="Repeat new password" />
            </div>
          </div>
          <div>
            <button className="btn btn-primary" onClick={handlePw}><Key size={15}/> Change Password</button>
          </div>
        </div>
      </div>
    </div>
  );
}
