import React, { useState, useEffect } from "react";
import { useAuth } from "../../context/AuthContext";
import { attendanceAPI } from "../../services/api";
import { Search, Download, Filter, CheckCircle, XCircle, Clock } from "lucide-react";
import toast from "react-hot-toast";

const STATUS_ICON = {
  present: <CheckCircle size={14} color="var(--success)" />,
  absent:  <XCircle    size={14} color="var(--error)"   />,
  late:    <Clock      size={14} color="var(--warning)"  />,
};

const MOCK_HISTORY = Array.from({ length: 20 }, (_, i) => {
  const d = new Date(); d.setDate(d.getDate() - i * 2);
  const statuses = ["present","present","present","late","absent"];
  return {
    id: i + 1,
    date: d.toISOString(),
    class_name: ["Machine Learning","Database Systems","Computer Networks","Operating Systems"][i % 4],
    subject_code: ["CS601","CS401","CS501","CS301"][i % 4],
    status: statuses[i % 5],
    time_marked: `${9 + (i % 4)}:${i % 2 === 0 ? "00" : "15"} AM`,
    confidence_score: (0.82 + Math.random() * 0.15).toFixed(2),
  };
});

export default function StudentHistory() {
  const { auth } = useAuth();
  const [records, setRecords] = useState(MOCK_HISTORY);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    if (!auth?.id) return;
    attendanceAPI.list({ student_id: auth.id, limit: 200 }).then((r) => { if (r.data?.length) setRecords(r.data); }).catch(() => {});
  }, [auth]);

  const filtered = records.filter((r) => {
    const q = search.toLowerCase();
    const matchSearch = r.class_name?.toLowerCase().includes(q) || r.subject_code?.toLowerCase().includes(q);
    const matchFilter = statusFilter === "all" || r.status === statusFilter;
    return matchSearch && matchFilter;
  });

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Attendance History</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
            {records.length} records found
          </p>
        </div>
        <button className="btn btn-outline btn-sm" onClick={() => toast.success("Certificate download (demo)")}>
          <Download size={14}/> Download Certificate
        </button>
      </div>

      <div style={{ display: "flex", gap: "0.875rem", flexWrap: "wrap" }}>
        <div className="search-bar">
          <Search size={15}/>
          <input placeholder="Search subject…" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <div style={{ display: "flex", gap: "0.4rem" }}>
          {[["all","All"],["present","Present"],["late","Late"],["absent","Absent"]].map(([v,l]) => (
            <button key={v} onClick={() => setStatusFilter(v)} className="btn btn-outline btn-sm"
              style={{ background: statusFilter === v ? "var(--primary)" : undefined, color: statusFilter === v ? "#fff" : undefined, borderColor: statusFilter === v ? "var(--primary)" : undefined }}>
              {l}
            </button>
          ))}
        </div>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrapper">
          <table className="data-table">
            <thead><tr>
              <th>Date</th><th>Subject</th><th>Status</th><th>Time Marked</th><th>Confidence</th>
            </tr></thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.id}>
                  <td style={{ fontWeight: 600 }}>
                    {r.date ? new Date(r.date).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—"}
                  </td>
                  <td>
                    <div style={{ fontWeight: 600 }}>{r.class_name || r.subject_name}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "monospace" }}>{r.subject_code}</div>
                  </td>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                      {STATUS_ICON[r.status]}
                      <span className={`badge badge-${r.status === "present" ? "success" : r.status === "late" ? "warning" : "error"}`}>
                        {r.status}
                      </span>
                    </div>
                  </td>
                  <td style={{ color: "var(--text-muted)" }}>{r.time_marked || "—"}</td>
                  <td>
                    {r.confidence_score
                      ? <span style={{ fontFamily: "monospace", fontSize: "0.82rem", color: r.confidence_score >= 0.9 ? "var(--success)" : r.confidence_score >= 0.75 ? "var(--warning)" : "var(--error)" }}>
                          {(r.confidence_score * 100).toFixed(0)}%
                        </span>
                      : "—"
                    }
                  </td>
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
