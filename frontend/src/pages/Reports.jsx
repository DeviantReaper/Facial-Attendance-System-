import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from "recharts";
import { Download, Filter, Search, FileText, AlertTriangle } from "lucide-react";
import toast from "react-hot-toast";

const COLORS_PIE = ["#10b981", "#ef4444", "#f59e0b"];

const WEEKLY_DATA = [
  { day: "Mon", present: 38, absent: 4 },
  { day: "Tue", present: 42, absent: 0 },
  { day: "Wed", present: 35, absent: 7 },
  { day: "Thu", present: 40, absent: 2 },
  { day: "Fri", present: 30, absent: 12 },
];

const TREND_DATA = [
  { week: "W1", rate: 88 }, { week: "W2", rate: 91 }, { week: "W3", rate: 85 },
  { week: "W4", rate: 93 }, { week: "W5", rate: 78 }, { week: "W6", rate: 89 },
];

const PIE_DATA = [{ name: "Present", value: 78 }, { name: "Absent", value: 15 }, { name: "Late", value: 7 }];

const DEFAULTERS = [
  { name: "Rohit Kumar",    roll_no: "MUJ2022003", dept: "ECE", pct: 62, minimum: 75 },
  { name: "Ananya Patel",   roll_no: "MUJ2022006", dept: "CSE", pct: 68, minimum: 75 },
  { name: "Sumit Joshi",    roll_no: "MUJ2022009", dept: "ME",  pct: 71, minimum: 75 },
];

export default function Reports() {
  const [startDate, setStartDate] = useState("");
  const [endDate,   setEndDate]   = useState("");
  const [selClass,  setSelClass]  = useState("all");

  const handleExport = (type) => {
    toast.success(`Exporting ${type.toUpperCase()} report… (demo)`);
  };

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Attendance Reports</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
            Analytics, trends, and defaulters list
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button className="btn btn-outline" onClick={() => handleExport("excel")}>
            <Download size={15}/> Excel
          </button>
          <button className="btn btn-primary" onClick={() => handleExport("pdf")}>
            <FileText size={15}/> PDF Report
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card" style={{ padding: "1rem 1.25rem", display: "flex", gap: "1rem", alignItems: "flex-end", flexWrap: "wrap" }}>
        <div className="form-group" style={{ flex: 1, minWidth: 160 }}>
          <label className="form-label">From Date</label>
          <input className="form-input" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </div>
        <div className="form-group" style={{ flex: 1, minWidth: 160 }}>
          <label className="form-label">To Date</label>
          <input className="form-input" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
        </div>
        <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
          <label className="form-label">Class</label>
          <select className="form-select" value={selClass} onChange={(e) => setSelClass(e.target.value)}>
            <option value="all">All Classes</option>
            <option value="1">CS601 — Machine Learning</option>
            <option value="2">CS401 — Database Systems</option>
            <option value="3">CS501 — Computer Networks</option>
          </select>
        </div>
        <button className="btn btn-primary" style={{ flexShrink: 0 }}>
          <Filter size={15}/> Apply Filter
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid-4">
        {[
          { label: "Total Sessions",   value: "42",  color: "primary" },
          { label: "Avg Attendance",   value: "83%", color: "success" },
          { label: "Total Students",   value: "127", color: "accent" },
          { label: "Defaulters",       value: "8",   color: "error" },
        ].map((s) => (
          <div key={s.label} className="card" style={{ textAlign: "center", padding: "1.25rem" }}>
            <div style={{ fontSize: "2rem", fontWeight: 800, color: `var(--${s.color})` }}>{s.value}</div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Daily Attendance (This Week)</div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={WEEKLY_DATA} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="day" stroke="var(--text-muted)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="present" name="Present" fill="var(--success)" radius={[4,4,0,0]} stackId="a" />
              <Bar dataKey="absent"  name="Absent"  fill="var(--error)"   radius={[4,4,0,0]} stackId="a" />
              <Legend iconType="circle" iconSize={10} formatter={(v) => <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{v}</span>} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Weekly Attendance Trend</div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={TREND_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="week" stroke="var(--text-muted)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis unit="%" stroke="var(--text-muted)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} domain={[70, 100]} />
              <Tooltip formatter={(v) => [`${v}%`, "Rate"]} contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
              <Line type="monotone" dataKey="rate" stroke="var(--primary)" strokeWidth={2.5} dot={{ fill: "var(--accent)", r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Pie + Defaulters */}
      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: "1.25rem" }}>
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div className="card-title">Overall Distribution</div>
          <ResponsiveContainer width="100%" height={170}>
            <PieChart>
              <Pie data={PIE_DATA} cx="50%" cy="50%" innerRadius={45} outerRadius={70} dataKey="value" paddingAngle={3}>
                {PIE_DATA.map((_, i) => <Cell key={i} fill={COLORS_PIE[i]} />)}
              </Pie>
              <Legend iconType="circle" iconSize={10} formatter={(v) => <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{v}</span>} />
              <Tooltip formatter={(v) => [`${v}%`]} contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card" style={{ padding: 0 }}>
          <div style={{ padding: "1rem 1.25rem", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div className="card-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <AlertTriangle size={16} color="var(--error)" /> Defaulters List
              </div>
              <div className="card-subtitle">Students below minimum 75% attendance</div>
            </div>
            <button className="btn btn-outline btn-sm" onClick={() => handleExport("defaulters")}>
              <Download size={13}/> Export
            </button>
          </div>
          <div className="table-wrapper">
            <table className="data-table">
              <thead><tr>
                <th>Student</th><th>Department</th><th>Attendance</th><th>Shortfall</th>
              </tr></thead>
              <tbody>
                {DEFAULTERS.map((d) => (
                  <tr key={d.roll_no}>
                    <td>
                      <div style={{ fontWeight: 600 }}>{d.name}</div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "monospace" }}>{d.roll_no}</div>
                    </td>
                    <td>{d.dept}</td>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                        <div className="progress-track" style={{ flex: 1, height: 6 }}>
                          <div className="progress-fill error" style={{ width: `${d.pct}%` }} />
                        </div>
                        <span style={{ fontWeight: 700, color: "var(--error)", fontSize: "0.85rem", minWidth: 40 }}>{d.pct}%</span>
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-error">-{d.minimum - d.pct}%</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
