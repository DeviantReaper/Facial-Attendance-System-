import React, { useState, useEffect } from "react";
import { useAuth } from "../../context/AuthContext";
import { attendanceAPI } from "../../services/api";
import { AlertTriangle, CheckCircle, Clock, BookOpen, TrendingUp } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = ["#10b981","#ef4444","#f59e0b"];

const MOCK_STATS = [
  { class_id: 1, subject_name: "Machine Learning",   subject_code: "CS601", color: "#f97316", present: 36, late: 2, absent: 4, total: 42, attendance_pct: 90.5, min_required: 75, is_at_risk: false },
  { class_id: 2, subject_name: "Database Systems",   subject_code: "CS401", color: "#1e3a5f", present: 28, late: 1, absent: 11, total: 40, attendance_pct: 72.5, min_required: 75, is_at_risk: true  },
  { class_id: 3, subject_name: "Computer Networks",  subject_code: "CS501", color: "#10b981", present: 32, late: 3, absent: 5, total: 40, attendance_pct: 87.5, min_required: 75, is_at_risk: false },
  { class_id: 4, subject_name: "Operating Systems",  subject_code: "CS301", color: "#8b5cf6", present: 30, late: 2, absent: 8, total: 40, attendance_pct: 80.0, min_required: 75, is_at_risk: false },
];

const MOCK_CALENDAR = (() => {
  const days = [];
  const now = new Date();
  for (let i = 29; i >= 0; i--) {
    const d = new Date(now); d.setDate(now.getDate() - i);
    const r = Math.random();
    days.push({ date: d, status: i < 2 ? "future" : r > 0.85 ? "absent" : r > 0.75 ? "late" : "present" });
  }
  return days;
})();

function DonutChart({ pct, color = "#10b981", size = 100 }) {
  const data = [{ value: pct }, { value: 100 - pct }];
  return (
    <div className="donut-wrapper" style={{ width: size, height: size }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius="60%" outerRadius="80%" dataKey="value" startAngle={90} endAngle={-270}>
            <Cell fill={color} />
            <Cell fill="var(--border)" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="donut-center-text">
        <div style={{ fontSize: size * 0.18, fontWeight: 800, color }}>{pct}%</div>
      </div>
    </div>
  );
}

export default function StudentDashboard() {
  const { auth } = useAuth();
  const [stats, setStats] = useState(MOCK_STATS);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!auth?.id) return;
    attendanceAPI.studentStats(auth.id).then((r) => { if (r.data?.length) setStats(r.data); }).catch(() => {});
  }, [auth]);

  const atRisk = stats.filter((s) => s.is_at_risk);
  const overallPct = stats.length ? Math.round(stats.reduce((s, x) => s + x.attendance_pct, 0) / stats.length) : 0;
  const overall = [{ name: "Present", value: overallPct }, { name: "Absent", value: 100 - overallPct }];

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Header */}
      <div>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Welcome back, {auth?.name?.split(" ")[0]} 👋</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
          {new Date().toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long" })}
        </p>
      </div>

      {/* Alerts */}
      {atRisk.map((s) => (
        <div key={s.class_id} style={{
          display: "flex", alignItems: "center", gap: "0.875rem",
          padding: "0.875rem 1.25rem", borderRadius: "var(--radius)",
          background: "var(--error-bg)", border: "1px solid rgba(239,68,68,0.3)",
        }}>
          <AlertTriangle size={18} color="var(--error)" flexShrink={0} />
          <div>
            <div style={{ fontWeight: 700, color: "var(--error)", fontSize: "0.875rem" }}>
              ⚠ Low Attendance Warning — {s.subject_name}
            </div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Your attendance is <strong>{s.attendance_pct}%</strong> — minimum required is <strong>{s.min_required}%</strong>. You need to attend more classes.
            </div>
          </div>
        </div>
      ))}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: "1.25rem" }}>
        {/* Subjects */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="card-title" style={{ padding: "0 0.25rem" }}>Subject Attendance</div>
          {stats.map((s) => (
            <div key={s.class_id} className="card" style={{ display: "flex", gap: "1.25rem", alignItems: "center" }}>
              <DonutChart pct={Math.round(s.attendance_pct)} color={s.is_at_risk ? "#ef4444" : s.color} size={80} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <div style={{ fontWeight: 700 }}>{s.subject_name}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "monospace" }}>{s.subject_code}</div>
                  </div>
                  {s.is_at_risk
                    ? <span className="badge badge-error">⚠ At Risk</span>
                    : <span className="badge badge-success">✓ Good</span>
                  }
                </div>
                <div style={{ display: "flex", gap: "1rem", marginTop: "0.75rem", fontSize: "0.78rem" }}>
                  {[["Present", s.present, "success"], ["Late", s.late, "warning"], ["Absent", s.absent, "error"]].map(([l, v, t]) => (
                    <div key={l} style={{ textAlign: "center" }}>
                      <div style={{ fontWeight: 800, color: `var(--${t})`, fontSize: "1.1rem" }}>{v}</div>
                      <div style={{ color: "var(--text-muted)" }}>{l}</div>
                    </div>
                  ))}
                </div>
                <div className="progress-track" style={{ marginTop: "0.625rem" }}>
                  <div className="progress-fill" style={{
                    width: `${s.attendance_pct}%`,
                    background: s.is_at_risk ? "var(--error)" : s.color,
                  }} />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Right: Overall + Calendar */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {/* Overall donut */}
          <div className="card" style={{ textAlign: "center" }}>
            <div className="card-title" style={{ marginBottom: "1rem" }}>Overall Attendance</div>
            <DonutChart pct={overallPct} color={overallPct >= 75 ? "#10b981" : "#ef4444"} size={130} />
            <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginTop: "0.875rem" }}>
              Across {stats.length} subjects
            </p>
          </div>

          {/* Calendar heatmap */}
          <div className="card">
            <div className="card-title" style={{ marginBottom: "0.875rem" }}>Last 30 Days</div>
            <div className="calendar-grid">
              {["S","M","T","W","T","F","S"].map((d) => (
                <div key={d} style={{ textAlign: "center", fontSize: "0.65rem", color: "var(--text-muted)", fontWeight: 700, paddingBottom: 4 }}>{d}</div>
              ))}
              {MOCK_CALENDAR.map((d, i) => (
                <div key={i} className={`calendar-day cal-${d.status}`} title={`${d.date.toLocaleDateString()} — ${d.status}`}>
                  {d.date.getDate()}
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.875rem", fontSize: "0.72rem", flexWrap: "wrap" }}>
              {[["cal-present","Present"],["cal-late","Late"],["cal-absent","Absent"]].map(([cls, l]) => (
                <div key={l} style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                  <div className={`calendar-day ${cls}`} style={{ width: 14, height: 14, borderRadius: 3, fontSize: 0 }} />
                  <span style={{ color: "var(--text-muted)" }}>{l}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
