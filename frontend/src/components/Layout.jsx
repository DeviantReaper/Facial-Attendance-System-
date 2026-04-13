import React from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import {
  LayoutDashboard, BookOpen, Users, Camera, BarChart3,
  Settings, Plug, Sun, Moon, LogOut, Bell, ChevronDown,
  ClipboardList,
} from "lucide-react";

const NAV = [
  {
    section: "Main",
    items: [
      { to: "/",              icon: <LayoutDashboard size={17}/>, label: "Dashboard" },
      { to: "/classes",       icon: <BookOpen size={17}/>,         label: "Classes" },
      { to: "/students",      icon: <Users size={17}/>,            label: "Students" },
      { to: "/face-enrollment",icon: <Camera size={17}/>,          label: "Face Enrollment" },
    ],
  },
  {
    section: "Analytics",
    items: [
      { to: "/reports",  icon: <BarChart3 size={17}/>,    label: "Reports" },
      { to: "/history",  icon: <ClipboardList size={17}/>,label: "Attendance History" },
    ],
  },
  {
    section: "System",
    items: [
      { to: "/integrations", icon: <Plug size={17}/>,     label: "Integrations" },
      { to: "/settings",     icon: <Settings size={17}/>, label: "Settings" },
    ],
  },
];

export default function Layout() {
  const { auth, logout } = useAuth();
  const { theme, toggle } = useTheme();
  const navigate = useNavigate();

  return (
    <div className="app-layout" data-theme={theme === "dark" ? "dark" : undefined}>
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        {/* Brand */}
        <div className="sidebar-brand">
          <h1>FaceAttend<span className="accent-dot">.</span></h1>
          <div className="brand-sub">Attendance Portal</div>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {NAV.map((section) => (
            <div key={section.section}>
              <div className="sidebar-section-label">{section.section}</div>
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === "/"}
                  className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
                >
                  {item.icon}
                  {item.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer">
          <div style={{
            display: "flex", alignItems: "center", gap: "0.75rem",
            padding: "0.75rem", borderRadius: "var(--radius)",
            background: "rgba(255,255,255,0.07)", marginBottom: "0.5rem",
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: "50%",
              background: "var(--accent)", color: "#fff",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontWeight: 700, fontSize: "0.9rem", flexShrink: 0,
            }}>
              {auth?.name?.[0]?.toUpperCase() || "T"}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "#fff", truncate: true }}>
                {auth?.name || "Teacher"}
              </div>
              <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", textTransform: "capitalize" }}>
                {auth?.role || "faculty"}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button onClick={toggle} className="nav-item" style={{ flex: 1, justifyContent: "center", fontSize: "0.78rem" }}>
              {theme === "dark" ? <><Sun size={14}/> Light</> : <><Moon size={14}/> Dark</>}
            </button>
            <button onClick={logout} className="nav-item" style={{ flex: 1, justifyContent: "center", color: "#fca5a5", fontSize: "0.78rem" }}>
              <LogOut size={14}/> Logout
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="main-content">
        <div className="page-content animate-fade-in">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
