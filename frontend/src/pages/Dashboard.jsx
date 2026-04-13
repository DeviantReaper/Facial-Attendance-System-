import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Users, BookOpen, Camera, TrendingUp, TrendingDown,
  ArrowRight, AlertTriangle, CheckCircle, Clock, Plus
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, PieChart, Pie, Cell, Legend,
} from "recharts";
import { useAuth } from "../context/AuthContext";
import { classesAPI, attendanceAPI } from "../services/api";

const COLORS = ["#10b981", "#ef4444", "#f59e0b"];

const MOCK_WEEKLY = [
  { day: "Mon", rate: 91 }, { day: "Tue", rate: 86 }, { day: "Wed", rate: 94 },
  { day: "Thu", rate: 88 }, { day: "Fri", rate: 78 }, { day: "Sat", rate: 72 },
];

const MOCK_PIE = [
  { name: "Present", value: 78 },
  { name: "Absent", value: 15 },
  { name: "Late", value: 7 },
];

const MOCK_CLASSES = [
  { id: 1, subject_name: "Machine Learning", subject_code: "CS601", room_number: "LHC-3", enrolled_count: 42, color: "#f97316", schedule: { day: "Mon/Wed", time: "10:00 AM" } },
  { id: 2, subject_name: "Database Systems", subject_code: "CS401", room_number: "LHC-1", enrolled_count: 38, color: "#1e3a5f", schedule: { day: "Tue/Thu", time: "2:00 PM" } },
  { id: 3, subject_name: "Computer Networks", subject_code: "CS501", room_number: "LHC-2", enrolled_count: 35, color: "#10b981", schedule: { day: "Mon/Fri", time: "9:00 AM" } },
];

const ACTIVITY = [
  { icon: <CheckCircle size={15} color="#10b981"/>, text: "Arjun Mehta marked present — CS601", time: "2 min ago" },
  { icon: <AlertTriangle size={15} color="#f59e0b"/>, text: "Priya Singh attendance below 75% in CS401", time: "15 min ago" },
  { icon: <Camera size={15} color="#1e3a5f"/>, text: "17 students enrolled via face scan", time: "1 hr ago" },
  { icon: <Clock size={15} color="#ef4444"/>, text: "Rohit Kumar marked late — CS501", time: "2 hr ago" },
];

export default function Dashboard() {
  const { auth } = useAuth();
  const navigate = useNavigate();
  const [classes, setClasses] = useState(MOCK_CLASSES);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    classesAPI.list().then((r) => {
      if (r.data?.length) setClasses(r.data);
    }).catch(() => {/* use mock */});
  }, []);

  const totalStudents = classes.reduce((s, c) => s + (c.enrolled_count || 0), 0);

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.4rem", fontWeight: 800, letterSpacing: "-0.02em" }}>
            Good morning, {auth?.name?.split(" ")[0] || "Professor"} 👋
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: 4 }}>
            {new Date().toLocaleDateString("en-IN", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button className="btn btn-outline" onClick={() => navigate("/students")}>
            <Plus size={15}/> Add Student
          </button>
          <button className="btn btn-primary" onClick={() => navigate("/classes")}>
            <Camera size={15}/> Start Attendance
          </button>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid-4">
        {[
          { label: "Total Students", value: totalStudents, icon: <Users size={22}/>, type: "primary", change: "+5 this month" },
          { label: "Classes Active", value: classes.length, icon: <BookOpen size={22}/>, type: "accent", change: "Live today" },
          { label: "Present Today", value: "78%", icon: <CheckCircle size={22}/>, type: "success", change: "↑ 3% vs yesterday" },
          { label: "Pending Enrollment", value: 4, icon: <Clock size={22}/>, type: "warning", change: "Needs review" },
        ].map((s) => (
          <div key={s.label} className="stat-card">
            <div className={`stat-icon ${s.type}`}>{s.icon}</div>
            <div>
              <div className="stat-value">{s.value}</div>
              <div className="stat-label">{s.label}</div>
              <div className={`stat-change ${s.change.startsWith("↑") ? "up" : ""}`} style={{ fontSize: "0.72rem", marginTop: 2, color: "var(--text-muted)" }}>
                {s.change}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Main content grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: "1.25rem" }}>

        {/* Left column */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Weekly bar chart */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Weekly Attendance Rate</div>
                <div className="card-subtitle">This week's daily attendance %</div>
              </div>
              <button className="btn btn-outline btn-sm" onClick={() => navigate("/reports")}>View Report</button>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={MOCK_WEEKLY} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="day" stroke="var(--text-muted)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis unit="%" stroke="var(--text-muted)" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} domain={[60, 100]} />
                <Tooltip formatter={(v) => [`${v}%`, "Attendance"]}
                  contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="rate" radius={[6, 6, 0, 0]}>
                  {MOCK_WEEKLY.map((_, i) => (
                    <Cell key={i} fill={_.rate >= 90 ? "#10b981" : _.rate >= 80 ? "#f97316" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Quick start attendance */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Quick Start Attendance</div>
                <div className="card-subtitle">Select a class to open the face recognition session</div>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.625rem" }}>
              {classes.map((cls) => (
                <button key={cls.id} onClick={() => navigate(`/classes/${cls.id}`)}
                  style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "0.875rem 1rem", border: "1px solid var(--border)", borderRadius: "var(--radius)",
                    background: "var(--bg-surface)", cursor: "pointer", textAlign: "left",
                    transition: "var(--transition)", fontFamily: "inherit",
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--primary)"; e.currentTarget.style.background = "var(--bg-card)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.background = "var(--bg-surface)"; }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "0.875rem" }}>
                    <div style={{ width: 4, height: 40, borderRadius: 4, background: cls.color || "var(--primary)", flexShrink: 0 }} />
                    <div>
                      <div style={{ fontWeight: 700, fontSize: "0.9rem", color: "var(--text)" }}>
                        {cls.subject_code} — {cls.subject_name}
                      </div>
                      <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 2 }}>
                        {cls.room_number && `Room ${cls.room_number} · `}
                        {cls.enrolled_count} students enrolled
                        {cls.schedule && ` · ${cls.schedule.day} ${cls.schedule.time}`}
                      </div>
                    </div>
                  </div>
                  <ArrowRight size={16} color="var(--text-muted)" />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right column */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Pie chart */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Today's Summary</div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={MOCK_PIE} cx="50%" cy="50%" innerRadius={50} outerRadius={75}
                  dataKey="value" paddingAngle={3}>
                  {MOCK_PIE.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Legend iconType="circle" iconSize={10} formatter={(v) => <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{v}</span>} />
                <Tooltip formatter={(v) => [`${v}%`]}
                  contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Activity feed */}
          <div className="card" style={{ flex: 1 }}>
            <div className="card-header">
              <div className="card-title">Recent Activity</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {ACTIVITY.map((a, i) => (
                <div key={i} style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
                  <div style={{ marginTop: 2, flexShrink: 0 }}>{a.icon}</div>
                  <div>
                    <div style={{ fontSize: "0.82rem", color: "var(--text)" }}>{a.text}</div>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 2 }}>{a.time}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
