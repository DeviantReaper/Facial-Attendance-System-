import React, { useState, useEffect } from "react";
import { Search, Download, Filter } from "lucide-react";
import { attendanceAPI } from "../services/api";
import toast from "react-hot-toast";

const MOCK_RECORDS = Array.from({ length: 25 }, (_, i) => {
  const d = new Date(); d.setDate(d.getDate() - i);
  const statuses = ["present","present","present","late","absent"];
  return {
    id: i + 1, student_name: ["Arjun Mehta","Priya Singh","Rohit Kumar","Sneha Gupta","Vikas Sharma"][i % 5],
    roll_no: `MUJ202200${(i % 5) + 1}`, class_name: ["Machine Learning","Database Systems","Computer Networks"][i % 3],
    subject_code: ["CS601","CS401","CS501"][i % 3],
    date: d.toISOString(), status: statuses[i % 5],
    confidence_score: (0.78 + Math.random() * 0.2).toFixed(2),
    is_manual_override: i % 7 === 0,
  };
});

export default function History() {
  const [records, setRecords] = useState(MOCK_RECORDS);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate]   = useState("");

  useEffect(() => {
    attendanceAPI.list({ limit: 200 }).then((r) => { if (r.data?.length) setRecords(r.data); }).catch(() => {});
  }, []);

  const filtered = records.filter((r) => {
    const q = search.toLowerCase();
    return (r.student_name?.toLowerCase().includes(q) || r.roll_no?.toLowerCase().includes(q) || r.class_name?.toLowerCase().includes(q)) &&
           (statusFilter === "all" || r.status === statusFilter);
  });

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Attendance History</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>{filtered.length} records</p>
        </div>
        <button className="btn btn-outline btn-sm" onClick={() => toast.success("Export started (demo)")}>
          <Download size={14}/> Export
        </button>
      </div>

      <div style={{ display: "flex", gap: "0.875rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <div className="search-bar">
          <Search size={15}/>
          <input placeholder="Search student, class…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ minWidth: 200 }} />
        </div>
        <div style={{ display: "flex", gap: "0.4rem" }}>
          {[["all","All"],["present","Present"],["late","Late"],["absent","Absent"]].map(([v,l]) => (
            <button key={v} onClick={() => setStatusFilter(v)} className="btn btn-outline btn-sm"
              style={{ background: statusFilter === v ? "var(--primary)" : undefined, color: statusFilter === v ? "#fff" : undefined, borderColor: statusFilter === v ? "var(--primary)" : undefined }}>
              {l}
            </button>
          ))}
        </div>
        <input className="form-input" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={{ width: 160 }} />
        <input className="form-input" type="date" value={endDate}   onChange={(e) => setEndDate(e.target.value)}   style={{ width: 160 }} />
      </div>

      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrapper">
          <table className="data-table">
            <thead><tr>
              <th>Date</th><th>Student</th><th>Class</th><th>Status</th><th>Confidence</th><th>Override</th>
            </tr></thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.id}>
                  <td style={{ fontWeight: 600, whiteSpace: "nowrap" }}>
                    {r.date ? new Date(r.date).toLocaleDateString("en-IN", { day: "numeric", month: "short" }) : "—"}
                  </td>
                  <td>
                    <div style={{ fontWeight: 600 }}>{r.student_name}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "monospace" }}>{r.roll_no}</div>
                  </td>
                  <td>
                    <div>{r.class_name}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{r.subject_code}</div>
                  </td>
                  <td>
                    <span className={`badge badge-${r.status === "present" ? "success" : r.status === "late" ? "warning" : "error"}`}>
                      {r.status}
                    </span>
                  </td>
                  <td>
                    {r.confidence_score ? (
                      <span style={{ fontFamily: "monospace", fontSize: "0.82rem", color: r.confidence_score >= 0.9 ? "var(--success)" : r.confidence_score >= 0.75 ? "var(--warning)" : "var(--error)" }}>
                        {(r.confidence_score * 100).toFixed(0)}%
                      </span>
                    ) : "—"}
                  </td>
                  <td>{r.is_manual_override ? <span className="badge badge-warning">Manual</span> : <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>Auto</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>No records found</div>
          )}
        </div>
      </div>
    </div>
  );
}
