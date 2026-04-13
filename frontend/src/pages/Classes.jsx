import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, Search, BookOpen, Edit2, Trash2, Users, Camera, X, Save } from "lucide-react";
import toast from "react-hot-toast";
import { classesAPI } from "../services/api";

const COLORS = ["#1e3a5f","#f97316","#10b981","#3b82f6","#8b5cf6","#ec4899","#f59e0b","#06b6d4"];

const MOCK_CLASSES = [
  { id: 1, subject_name: "Machine Learning", subject_code: "CS601", room_number: "LHC-3", enrolled_count: 42, color: "#f97316", schedule: { day: "Mon/Wed", time: "10:00 AM", duration: 60 }, min_attendance_pct: 75 },
  { id: 2, subject_name: "Database Systems", subject_code: "CS401", room_number: "LHC-1", enrolled_count: 38, color: "#1e3a5f", schedule: { day: "Tue/Thu", time: "2:00 PM", duration: 60 }, min_attendance_pct: 75 },
  { id: 3, subject_name: "Computer Networks", subject_code: "CS501", room_number: "LHC-2", enrolled_count: 35, color: "#10b981", schedule: { day: "Mon/Fri", time: "9:00 AM", duration: 90 }, min_attendance_pct: 75 },
];

const BLANK = { subject_name: "", subject_code: "", room_number: "", color: "#1e3a5f", min_attendance_pct: 75, late_threshold_minutes: 10, schedule: { day: "", time: "", duration: 60 } };

export default function Classes() {
  const navigate = useNavigate();
  const [classes, setClasses] = useState(MOCK_CLASSES);
  const [search, setSearch] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState(BLANK);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    classesAPI.list().then((r) => { if (r.data?.length) setClasses(r.data); }).catch(() => {});
  }, []);

  const filtered = classes.filter((c) =>
    c.subject_name.toLowerCase().includes(search.toLowerCase()) ||
    c.subject_code.toLowerCase().includes(search.toLowerCase())
  );

  const openCreate = () => { setEditing(null); setForm(BLANK); setShowModal(true); };
  const openEdit = (cls) => { setEditing(cls); setForm({ ...cls, schedule: cls.schedule || BLANK.schedule }); setShowModal(true); };
  const closeModal = () => { setShowModal(false); setEditing(null); };

  const handleSave = async () => {
    if (!form.subject_name || !form.subject_code) return toast.error("Subject name and code are required");
    setSaving(true);
    try {
      if (editing) {
        await classesAPI.update(editing.id, form);
        setClasses((p) => p.map((c) => c.id === editing.id ? { ...c, ...form } : c));
        toast.success("Class updated successfully");
      } else {
        const res = await classesAPI.create(form);
        setClasses((p) => [...p, { ...form, id: res.data.id, enrolled_count: 0 }]);
        toast.success("Class created successfully");
      }
      closeModal();
    } catch {
      // Optimistic update in mock mode
      if (editing) {
        setClasses((p) => p.map((c) => c.id === editing.id ? { ...c, ...form } : c));
      } else {
        setClasses((p) => [...p, { ...form, id: Date.now(), enrolled_count: 0 }]);
      }
      toast.success(editing ? "Class updated" : "Class created");
      closeModal();
    }
    setSaving(false);
  };

  const handleDelete = async (cls) => {
    if (!confirm(`Delete "${cls.subject_name}"?`)) return;
    try { await classesAPI.delete(cls.id); } catch {}
    setClasses((p) => p.filter((c) => c.id !== cls.id));
    toast.success("Class deleted");
  };

  const F = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));
  const Fs = (key) => (e) => setForm((f) => ({ ...f, schedule: { ...f.schedule, [key]: e.target.value } }));

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Class Management</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
            {classes.length} classes · {classes.reduce((s, c) => s + (c.enrolled_count || 0), 0)} total students
          </p>
        </div>
        <button className="btn btn-primary" onClick={openCreate}><Plus size={16}/> Create Class</button>
      </div>

      {/* Search */}
      <div className="search-bar" style={{ maxWidth: 360 }}>
        <Search size={16} />
        <input placeholder="Search by name or code…" value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>

      {/* Class grid */}
      <div className="grid-3">
        {filtered.map((cls) => (
          <div key={cls.id} className="card" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {/* Color bar */}
            <div style={{ height: 4, borderRadius: 4, background: cls.color || "#1e3a5f", margin: "-1.5rem -1.5rem 0" }} />

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <div>
                <div style={{ fontWeight: 800, fontSize: "1rem", color: "var(--text)" }}>{cls.subject_name}</div>
                <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 2, fontFamily: "monospace" }}>
                  {cls.subject_code}
                </div>
              </div>
              <div style={{ display: "flex", gap: "0.4rem" }}>
                <button className="btn btn-ghost btn-sm" onClick={() => openEdit(cls)}><Edit2 size={14}/></button>
                <button className="btn btn-ghost btn-sm" onClick={() => handleDelete(cls)} style={{ color: "var(--error)" }}><Trash2 size={14}/></button>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.8rem" }}>
              <div style={{ color: "var(--text-muted)" }}>Room</div>
              <div style={{ fontWeight: 600 }}>{cls.room_number || "—"}</div>
              <div style={{ color: "var(--text-muted)" }}>Schedule</div>
              <div style={{ fontWeight: 600 }}>{cls.schedule?.day || "—"} {cls.schedule?.time || ""}</div>
              <div style={{ color: "var(--text-muted)" }}>Min Attendance</div>
              <div style={{ fontWeight: 600 }}>{cls.min_attendance_pct}%</div>
            </div>

            <div style={{ borderTop: "1px solid var(--border)", paddingTop: "0.875rem", display: "flex", gap: "0.625rem" }}>
              <button className="btn btn-outline btn-sm" style={{ flex: 1 }} onClick={() => navigate(`/classes/${cls.id}`)}>
                <Users size={14}/> {cls.enrolled_count} Students
              </button>
              <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={() => navigate(`/classes/${cls.id}`)}>
                <Camera size={14}/> Attendance
              </button>
            </div>
          </div>
        ))}

        {/* Empty card */}
        <div className="card" onClick={openCreate} style={{
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          border: "2px dashed var(--border)", cursor: "pointer", minHeight: 200, gap: "0.75rem",
          transition: "var(--transition)",
        }}
          onMouseEnter={(e) => e.currentTarget.style.borderColor = "var(--primary)"}
          onMouseLeave={(e) => e.currentTarget.style.borderColor = "var(--border)"}
        >
          <Plus size={28} color="var(--text-light)" />
          <div style={{ color: "var(--text-muted)", fontSize: "0.875rem", fontWeight: 600 }}>Create New Class</div>
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && closeModal()}>
          <div className="modal animate-slide-up">
            <div className="modal-header">
              <div className="modal-title">{editing ? "Edit Class" : "Create New Class"}</div>
              <button className="btn btn-ghost btn-sm" onClick={closeModal}><X size={18}/></button>
            </div>
            <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Subject Name *</label>
                  <input className="form-input" placeholder="Machine Learning" value={form.subject_name} onChange={F("subject_name")} />
                </div>
                <div className="form-group">
                  <label className="form-label">Subject Code *</label>
                  <input className="form-input" placeholder="CS601" value={form.subject_code} onChange={F("subject_code")} />
                </div>
              </div>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Room Number</label>
                  <input className="form-input" placeholder="LHC-3" value={form.room_number} onChange={F("room_number")} />
                </div>
                <div className="form-group">
                  <label className="form-label">Schedule — Days</label>
                  <input className="form-input" placeholder="Mon/Wed" value={form.schedule?.day || ""} onChange={Fs("day")} />
                </div>
              </div>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Start Time</label>
                  <input className="form-input" type="time" value={form.schedule?.time || ""} onChange={Fs("time")} />
                </div>
                <div className="form-group">
                  <label className="form-label">Min Attendance %</label>
                  <input className="form-input" type="number" min={0} max={100} value={form.min_attendance_pct}
                    onChange={(e) => setForm((f) => ({ ...f, min_attendance_pct: parseFloat(e.target.value) }))} />
                </div>
              </div>
              {/* Color picker */}
              <div className="form-group">
                <label className="form-label">Class Color</label>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {COLORS.map((c) => (
                    <button key={c} onClick={() => setForm((f) => ({ ...f, color: c }))} style={{
                      width: 28, height: 28, borderRadius: "50%", background: c, border: "none",
                      cursor: "pointer", outline: form.color === c ? `3px solid ${c}` : "none",
                      outlineOffset: 2,
                    }} />
                  ))}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-outline" onClick={closeModal}>Cancel</button>
              <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
                {saving ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Save size={15} />}
                {editing ? "Save Changes" : "Create Class"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
