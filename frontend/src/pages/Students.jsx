import React, { useState, useEffect } from "react";
import {
  Plus, Search, Filter, UserCheck, UserX, Edit2, Trash2,
  Download, Upload, Eye, Camera
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import { studentsAPI } from "../services/api";

const MOCK = [
  { id: 1, name: "Arjun Mehta",  roll_no: "MUJ2022001", email: "arjun@muj.edu",  department: "CSE", semester: 6, is_enrolled: true,  face_count: 3, enrollment_status: "approved" },
  { id: 2, name: "Priya Singh",  roll_no: "MUJ2022002", email: "priya@muj.edu",  department: "CSE", semester: 6, is_enrolled: true,  face_count: 5, enrollment_status: "approved" },
  { id: 3, name: "Rohit Kumar",  roll_no: "MUJ2022003", email: "rohit@muj.edu",  department: "ECE", semester: 4, is_enrolled: false, face_count: 0, enrollment_status: "pending" },
  { id: 4, name: "Sneha Gupta",  roll_no: "MUJ2022004", email: "sneha@muj.edu",  department: "CSE", semester: 6, is_enrolled: false, face_count: 0, enrollment_status: "approved" },
  { id: 5, name: "Vikas Sharma", roll_no: "MUJ2022005", email: "vikas@muj.edu",  department: "ME",  semester: 2, is_enrolled: true,  face_count: 4, enrollment_status: "approved" },
  { id: 6, name: "Ananya Patel", roll_no: "MUJ2022006", email: "ananya@muj.edu", department: "CSE", semester: 4, is_enrolled: false, face_count: 0, enrollment_status: "pending" },
];

const BLANK_FORM = { name: "", roll_no: "", email: "", phone: "", department: "", semester: "" };

export default function Students() {
  const navigate = useNavigate();
  const [students, setStudents] = useState(MOCK);
  const [search, setSearch] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState(BLANK_FORM);
  const [saving, setSaving] = useState(false);
  const [filter, setFilter] = useState("all"); // all | enrolled | not_enrolled

  useEffect(() => {
    studentsAPI.list({ limit: 100 }).then((r) => { if (r.data?.students?.length) setStudents(r.data.students); }).catch(() => {});
  }, []);

  const filtered = students.filter((s) => {
    const q = search.toLowerCase();
    const matchSearch = s.name.toLowerCase().includes(q) || s.roll_no.toLowerCase().includes(q) || s.email?.toLowerCase().includes(q);
    const matchFilter = filter === "all" || (filter === "enrolled" ? s.is_enrolled : !s.is_enrolled);
    return matchSearch && matchFilter;
  });

  const handleCreate = async () => {
    if (!form.name || !form.roll_no || !form.email) return toast.error("Name, roll number, and email are required");
    setSaving(true);
    try {
      const res = await studentsAPI.create({ ...form, semester: form.semester ? +form.semester : undefined });
      setStudents((p) => [...p, { ...form, id: res.data.id, is_enrolled: false, face_count: 0, enrollment_status: "approved" }]);
      toast.success("Student added successfully");
    } catch {
      setStudents((p) => [...p, { ...form, id: Date.now(), is_enrolled: false, face_count: 0, enrollment_status: "approved" }]);
      toast.success("Student added (demo mode)");
    }
    setShowModal(false);
    setForm(BLANK_FORM);
    setSaving(false);
  };

  const handleDelete = async (s) => {
    if (!confirm(`Delete "${s.name}"?`)) return;
    try { await studentsAPI.delete(s.id); } catch {}
    setStudents((p) => p.filter((x) => x.id !== s.id));
    toast.success("Student removed");
  };

  const handleCSV = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await studentsAPI.bulkImport(fd);
      toast.success(`Imported ${res.data.created} students`);
    } catch {
      toast.success("CSV import processed (demo mode)");
    }
    e.target.value = "";
  };

  const F = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const enrolled = students.filter((s) => s.is_enrolled).length;

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Student Management</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
            {students.length} students · {enrolled} face-enrolled
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <label className="btn btn-outline" style={{ cursor: "pointer" }}>
            <Upload size={15}/> Import CSV
            <input type="file" accept=".csv" style={{ display: "none" }} onChange={handleCSV} />
          </label>
          <button className="btn btn-primary" onClick={() => setShowModal(true)}>
            <Plus size={15}/> Add Student
          </button>
        </div>
      </div>

      {/* Filter bar */}
      <div style={{ display: "flex", gap: "0.875rem", alignItems: "center", flexWrap: "wrap" }}>
        <div className="search-bar">
          <Search size={15}/>
          <input placeholder="Search name, roll no, email…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ minWidth: 240 }} />
        </div>
        <div style={{ display: "flex", gap: "0.4rem" }}>
          {[["all","All"], ["enrolled","Face Enrolled"], ["not_enrolled","Not Enrolled"]].map(([v, l]) => (
            <button key={v} onClick={() => setFilter(v)} className="btn btn-outline btn-sm"
              style={{ background: filter === v ? "var(--primary)" : undefined, color: filter === v ? "#fff" : undefined, borderColor: filter === v ? "var(--primary)" : undefined }}>
              {l}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrapper">
          <table className="data-table">
            <thead><tr>
              <th>Student</th>
              <th>Roll No.</th>
              <th>Department</th>
              <th>Semester</th>
              <th>Face Status</th>
              <th>Enrollment</th>
              <th style={{ textAlign: "right" }}>Actions</th>
            </tr></thead>
            <tbody>
              {filtered.map((s) => (
                <tr key={s.id}>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <div style={{
                        width: 34, height: 34, borderRadius: "50%",
                        background: s.is_enrolled ? "rgba(16,185,129,0.15)" : "var(--border)",
                        color: s.is_enrolled ? "var(--success)" : "var(--text-muted)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontWeight: 700, fontSize: "0.82rem", flexShrink: 0,
                      }}>{s.name[0]}</div>
                      <div>
                        <div style={{ fontWeight: 600 }}>{s.name}</div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{s.email}</div>
                      </div>
                    </div>
                  </td>
                  <td><code style={{ fontSize: "0.82rem" }}>{s.roll_no}</code></td>
                  <td>{s.department || "—"}</td>
                  <td>{s.semester ? `Sem ${s.semester}` : "—"}</td>
                  <td>
                    {s.is_enrolled
                      ? <span className="badge badge-success">✓ {s.face_count} descriptor(s)</span>
                      : <span className="badge badge-error">Not enrolled</span>
                    }
                  </td>
                  <td>
                    <span className={`badge ${s.enrollment_status === "approved" ? "badge-success" : s.enrollment_status === "pending" ? "badge-warning" : "badge-error"}`}>
                      {s.enrollment_status}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: "flex", gap: "0.4rem", justifyContent: "flex-end" }}>
                      <button className="btn btn-ghost btn-sm" onClick={() => navigate(`/face-enrollment/${s.id}`)} title="Face Enrollment">
                        <Camera size={14}/>
                      </button>
                      <button className="btn btn-ghost btn-sm" title="View profile">
                        <Eye size={14}/>
                      </button>
                      <button className="btn btn-ghost btn-sm" style={{ color: "var(--error)" }} onClick={() => handleDelete(s)} title="Delete">
                        <Trash2 size={14}/>
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>No students found</div>
          )}
        </div>
      </div>

      {/* Create modal */}
      {showModal && (
        <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && setShowModal(false)}>
          <div className="modal animate-slide-up">
            <div className="modal-header">
              <div className="modal-title">Add New Student</div>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowModal(false)}>✕</button>
            </div>
            <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Full Name *</label>
                  <input className="form-input" placeholder="Arjun Mehta" value={form.name} onChange={F("name")} />
                </div>
                <div className="form-group">
                  <label className="form-label">Roll Number *</label>
                  <input className="form-input" placeholder="MUJ2022001" value={form.roll_no} onChange={F("roll_no")} />
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Email Address *</label>
                <input className="form-input" type="email" placeholder="student@muj.edu" value={form.email} onChange={F("email")} />
              </div>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Phone</label>
                  <input className="form-input" placeholder="+91 9876543210" value={form.phone} onChange={F("phone")} />
                </div>
                <div className="form-group">
                  <label className="form-label">Department</label>
                  <select className="form-select" value={form.department} onChange={F("department")}>
                    <option value="">Select…</option>
                    {["CSE","ECE","ME","CE","EE","MBA","BCA"].map((d) => <option key={d}>{d}</option>)}
                  </select>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Semester</label>
                <select className="form-select" value={form.semester} onChange={F("semester")}>
                  <option value="">Select…</option>
                  {[1,2,3,4,5,6,7,8].map((s) => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div style={{ padding: "0.75rem", background: "var(--info-bg)", border: "1px solid rgba(59,130,246,0.2)", borderRadius: "var(--radius)", fontSize: "0.8rem", color: "var(--info)" }}>
                ℹ Default password: <code>student123</code>. Student should change it on first login.
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-outline" onClick={() => setShowModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleCreate} disabled={saving}>
                {saving ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Plus size={15}/>}
                Add Student
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
