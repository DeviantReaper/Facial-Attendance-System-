import React, { useState, useEffect } from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import {
  LayoutDashboard, BookOpen, Clock, User, Sun, Moon, LogOut
} from "lucide-react";

export default function StudentLayout() {
  const { auth, logout } = useAuth();
  const { theme, toggle } = useTheme();

  return (
    <div className="app-layout" data-theme={theme === "dark" ? "dark" : undefined}>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>FaceAttend<span className="accent-dot">.</span></h1>
          <div className="brand-sub">Student Portal</div>
        </div>

        <nav className="sidebar-nav">
          <div className="sidebar-section-label">My Portal</div>
          <NavLink to="/student" end className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
            <LayoutDashboard size={17}/> Dashboard
          </NavLink>
          <NavLink to="/student/subjects" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
            <BookOpen size={17}/> My Subjects
          </NavLink>
          <NavLink to="/student/history" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
            <Clock size={17}/> Attendance History
          </NavLink>
          <NavLink to="/student/profile" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
            <User size={17}/> My Profile
          </NavLink>
        </nav>

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
            }}>{auth?.name?.[0]?.toUpperCase() || "S"}</div>
            <div>
              <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "#fff" }}>{auth?.name || "Student"}</div>
              <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)" }}>Student</div>
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

      <div className="main-content">
        <div className="page-content animate-fade-in">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
