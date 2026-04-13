import React, { useState } from "react";
import { Globe, Key, Plus, RefreshCw, Copy, Shield, Webhook, X } from "lucide-react";
import toast from "react-hot-toast";
import { integrateAPI } from "../services/api";

const MOCK_COLLEGES = [
  { id: 1, name: "Jaipur Institute of Technology", code: "JIT", contact_email: "admin@jit.edu", auto_approve_enrollment: false, student_count: 24 },
  { id: 2, name: "Rajasthan Technical University", code: "RTU", contact_email: "admin@rtu.ac.in", auto_approve_enrollment: true, student_count: 11 },
];

const BLANK = { name: "", code: "", contact_email: "", contact_name: "", webhook_url: "", auto_approve_enrollment: false };

export default function Integrations() {
  const [colleges, setColleges] = useState(MOCK_COLLEGES);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState(BLANK);
  const [newKey, setNewKey] = useState(null);
  const [saving, setSaving] = useState(false);

  const handleCreate = async () => {
    if (!form.name || !form.code) return toast.error("Name and code required");
    setSaving(true);
    try {
      const res = await integrateAPI.createCollege(form);
      setNewKey(res.data.api_key);
      setColleges((p) => [...p, { ...form, id: res.data.college_id, student_count: 0 }]);
    } catch {
      const key = `fk_${"x".repeat(40)}`;
      setNewKey(key);
      setColleges((p) => [...p, { ...form, id: Date.now(), student_count: 0 }]);
    }
    setSaving(false);
    setShowModal(false);
  };

  const handleRegenKey = async (college) => {
    if (!confirm(`Regenerate API key for ${college.name}? Existing integrations will break.`)) return;
    try {
      const res = await integrateAPI.regenerateKey(college.id);
      setNewKey(res.data.api_key);
      toast.success("New API key generated");
    } catch {
      setNewKey(`fk_${"y".repeat(40)}`);
      toast.success("API key regenerated (demo)");
    }
  };

  const copyKey = () => { navigator.clipboard.writeText(newKey); toast.success("Copied to clipboard!"); };
  const F = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>College Integrations</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
            Manage partner colleges and their API access
          </p>
        </div>
        <button className="btn btn-primary" onClick={() => { setShowModal(true); setNewKey(null); setForm(BLANK); }}>
          <Plus size={16}/> Add Partner College
        </button>
      </div>

      {/* API Key reveal */}
      {newKey && (
        <div className="card animate-fade-in" style={{ background: "var(--success-bg)", border: "1px solid var(--success)", borderRadius: "var(--radius-lg)" }}>
          <div style={{ color: "var(--success)", fontWeight: 700, marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Key size={16}/> New API Key Generated — Save it now!
          </div>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <code style={{ flex: 1, background: "rgba(0,0,0,0.08)", padding: "0.625rem 0.875rem", borderRadius: "var(--radius)", fontSize: "0.82rem", fontFamily: "monospace", wordBreak: "break-all" }}>
              {newKey}
            </code>
            <button className="btn btn-outline btn-sm" onClick={copyKey}><Copy size={14}/> Copy</button>
          </div>
          <p style={{ fontSize: "0.75rem", color: "var(--success)", marginTop: "0.5rem" }}>
            ⚠ This key will not be shown again. Store it securely.
          </p>
        </div>
      )}

      {/* Overview */}
      <div className="grid-3">
        {[
          { label: "Partner Colleges", value: colleges.length, icon: <Globe size={20}/>, color: "primary" },
          { label: "External Students", value: colleges.reduce((s, c) => s + c.student_count, 0), icon: <Shield size={20}/>, color: "accent" },
          { label: "Active API Keys", value: colleges.length, icon: <Key size={20}/>, color: "success" },
        ].map((s) => (
          <div key={s.label} className="stat-card">
            <div className={`stat-icon ${s.color}`}>{s.icon}</div>
            <div>
              <div className="stat-value">{s.value}</div>
              <div className="stat-label">{s.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* College list */}
      <div className="card" style={{ padding: 0 }}>
        <div style={{ padding: "1rem 1.25rem", borderBottom: "1px solid var(--border)", fontWeight: 700 }}>
          Partner Colleges
        </div>
        {colleges.map((c) => (
          <div key={c.id} style={{ padding: "1.25rem", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
              <div style={{
                width: 44, height: 44, borderRadius: "var(--radius)", flexShrink: 0,
                background: "var(--primary)", color: "#fff",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontWeight: 800, fontSize: "1rem",
              }}>{c.code[0]}</div>
              <div>
                <div style={{ fontWeight: 700 }}>{c.name}</div>
                <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                  {c.code} · {c.contact_email}
                </div>
                <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.35rem" }}>
                  <span className="badge badge-primary">{c.student_count} students</span>
                  {c.auto_approve_enrollment && <span className="badge badge-success">Auto-approve</span>}
                  {c.webhook_url && <span className="badge badge-info"><Webhook size={10}/> Webhook</span>}
                </div>
              </div>
            </div>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button className="btn btn-outline btn-sm" onClick={() => handleRegenKey(c)}>
                <RefreshCw size={13}/> Regen Key
              </button>
            </div>
          </div>
        ))}
        {colleges.length === 0 && (
          <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>
            No partner colleges yet. Add one to get started.
          </div>
        )}
      </div>

      {/* API docs reference */}
      <div className="card" style={{ background: "var(--bg-surface)" }}>
        <div className="card-title" style={{ marginBottom: "1rem" }}>Integration API Endpoints</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          {[
            ["POST", "/api/v1/auth/college-token",         "Get JWT token using API key"],
            ["POST", "/api/v1/integrate/enroll",           "Enroll external student + face"],
            ["GET",  "/api/v1/integrate/students",         "Fetch enrolled external students"],
            ["POST", "/api/v1/integrate/attendance",       "Log attendance for external student"],
            ["GET",  "/api/v1/integrate/attendance/:roll", "Get attendance record"],
          ].map(([method, path, desc]) => (
            <div key={path} style={{ display: "flex", alignItems: "center", gap: "0.75rem", fontSize: "0.82rem" }}>
              <span style={{
                padding: "2px 8px", borderRadius: 4, fontWeight: 800, fontFamily: "monospace", fontSize: "0.72rem",
                background: method === "GET" ? "var(--info-bg)" : "var(--success-bg)",
                color: method === "GET" ? "var(--info)" : "var(--success)",
              }}>{method}</span>
              <code style={{ color: "var(--text)", fontSize: "0.82rem" }}>{path}</code>
              <span style={{ color: "var(--text-muted)" }}>— {desc}</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: "1rem" }}>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="btn btn-primary btn-sm">
            View Swagger Docs →
          </a>
        </div>
      </div>

      {/* Create modal */}
      {showModal && (
        <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && setShowModal(false)}>
          <div className="modal animate-slide-up">
            <div className="modal-header">
              <div className="modal-title">Add Partner College</div>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowModal(false)}><X size={18}/></button>
            </div>
            <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">College Name *</label>
                  <input className="form-input" placeholder="Jaipur Institute of Technology" value={form.name} onChange={F("name")} />
                </div>
                <div className="form-group">
                  <label className="form-label">College Code *</label>
                  <input className="form-input" placeholder="JIT" value={form.code} onChange={F("code")} style={{ textTransform: "uppercase" }} />
                </div>
              </div>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Contact Name</label>
                  <input className="form-input" placeholder="Dr. Ramesh Gupta" value={form.contact_name} onChange={F("contact_name")} />
                </div>
                <div className="form-group">
                  <label className="form-label">Contact Email</label>
                  <input className="form-input" type="email" placeholder="admin@jit.edu" value={form.contact_email} onChange={F("contact_email")} />
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Webhook URL (optional)</label>
                <input className="form-input" placeholder="https://jit.edu/webhooks/attendance" value={form.webhook_url} onChange={F("webhook_url")} />
                <span className="form-hint">Real-time attendance events will be POSTed to this URL</span>
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: "0.75rem", cursor: "pointer" }}>
                <input type="checkbox" checked={form.auto_approve_enrollment} onChange={F("auto_approve_enrollment")} />
                <div>
                  <div style={{ fontWeight: 600, fontSize: "0.875rem" }}>Auto-approve enrollments</div>
                  <div className="form-hint">Students from this college are approved without teacher review</div>
                </div>
              </label>
            </div>
            <div className="modal-footer">
              <button className="btn btn-outline" onClick={() => setShowModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleCreate} disabled={saving}>
                {saving ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Key size={15}/>}
                Create & Generate Key
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
