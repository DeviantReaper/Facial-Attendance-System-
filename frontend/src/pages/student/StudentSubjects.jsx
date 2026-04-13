import React, { useState, useEffect } from "react";
import { useAuth } from "../../context/AuthContext";
import { attendanceAPI } from "../../services/api";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

const MOCK_STATS = [
  { class_id: 1, subject_name: "Machine Learning",   subject_code: "CS601", color: "#f97316", present: 36, late: 2, absent: 4, total: 42, attendance_pct: 90.5, min_required: 75, is_at_risk: false },
  { class_id: 2, subject_name: "Database Systems",   subject_code: "CS401", color: "#1e3a5f", present: 28, late: 1, absent: 11, total: 40, attendance_pct: 72.5, min_required: 75, is_at_risk: true  },
  { class_id: 3, subject_name: "Computer Networks",  subject_code: "CS501", color: "#10b981", present: 32, late: 3, absent: 5, total: 40, attendance_pct: 87.5, min_required: 75, is_at_risk: false },
  { class_id: 4, subject_name: "Operating Systems",  subject_code: "CS301", color: "#8b5cf6", present: 30, late: 2, absent: 8, total: 40, attendance_pct: 80.0, min_required: 75, is_at_risk: false },
];

export default function StudentSubjects() {
  const { auth } = useAuth();
  const [stats, setStats] = useState(MOCK_STATS);

  useEffect(() => {
    if (!auth?.id) return;
    attendanceAPI.studentStats(auth.id).then((r) => { if (r.data?.length) setStats(r.data); }).catch(() => {});
  }, [auth]);

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>My Subjects</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
          Detailed breakdown per subject
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        {stats.map((s) => {
          const pieData = [
            { name: "Present", value: s.present },
            { name: "Late",    value: s.late },
            { name: "Absent",  value: s.absent },
          ];
          const colors = ["#10b981","#f59e0b","#ef4444"];
          return (
            <div key={s.class_id} className="card" style={{ padding: "1.25rem" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: "2rem", alignItems: "center" }}>
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                    <div>
                      <div style={{ fontWeight: 800, fontSize: "1.05rem" }}>{s.subject_name}</div>
                      <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "monospace", marginTop: 2 }}>{s.subject_code}</div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <div style={{ fontSize: "1.75rem", fontWeight: 800, color: s.is_at_risk ? "var(--error)" : s.color }}>
                        {s.attendance_pct}%
                      </div>
                      {s.is_at_risk
                        ? <span className="badge badge-error">⚠ At Risk</span>
                        : <span className="badge badge-success">✓ Good</span>
                      }
                    </div>
                  </div>

                  <div className="progress-track" style={{ marginBottom: "1rem" }}>
                    <div className="progress-fill" style={{ width: `${s.attendance_pct}%`, background: s.is_at_risk ? "var(--error)" : s.color }} />
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: "0.75rem", fontSize: "0.82rem" }}>
                    {[["Total Classes", s.total, "var(--text)"], ["Present", s.present, "var(--success)"], ["Late", s.late, "var(--warning)"], ["Absent", s.absent, "var(--error)"]].map(([l, v, c]) => (
                      <div key={l} style={{ textAlign: "center", padding: "0.75rem", background: "var(--bg-surface)", borderRadius: "var(--radius)" }}>
                        <div style={{ fontWeight: 800, fontSize: "1.2rem", color: c }}>{v}</div>
                        <div style={{ color: "var(--text-muted)", marginTop: 2 }}>{l}</div>
                      </div>
                    ))}
                  </div>

                  {s.is_at_risk && (
                    <div style={{ marginTop: "0.875rem", padding: "0.75rem", background: "var(--error-bg)", borderRadius: "var(--radius)", fontSize: "0.8rem", color: "var(--error)" }}>
                      ⚠ You need <strong>{Math.ceil(s.total * s.min_required / 100) - s.present}</strong> more present classes to reach the minimum {s.min_required}% requirement.
                    </div>
                  )}
                </div>

                <div>
                  <ResponsiveContainer width="100%" height={160}>
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" outerRadius={65} dataKey="value" paddingAngle={2}>
                        {pieData.map((_, i) => <Cell key={i} fill={colors[i]} />)}
                      </Pie>
                      <Tooltip formatter={(v, n) => [`${v} classes`, n]} contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
