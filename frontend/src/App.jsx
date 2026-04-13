import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "./context/ThemeContext";
import { AuthProvider, useAuth } from "./context/AuthContext";

// Layouts
import Layout from "./components/Layout";
import StudentLayout from "./components/StudentLayout";

// Auth pages
import LoginPage from "./pages/LoginPage";

// Teacher/Admin pages
import Dashboard from "./pages/Dashboard";
import Classes from "./pages/Classes";
import ClassDetail from "./pages/ClassDetail";
import Students from "./pages/Students";
import FaceEnrollment from "./pages/FaceEnrollment";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import Integrations from "./pages/Integrations";
import History from "./pages/History";

// Student pages
import StudentDashboard from "./pages/student/StudentDashboard";
import StudentSubjects from "./pages/student/StudentSubjects";
import StudentHistory from "./pages/student/StudentHistory";
import StudentEnrollment from "./pages/student/StudentEnrollment";
import StudentProfile from "./pages/student/StudentProfile";

function AppRoutes() {
  const { auth } = useAuth();

  if (!auth) {
    return (
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/enroll" element={<StudentEnrollment />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    );
  }

  if (auth.role === "student") {
    return (
      <Routes>
        <Route path="/student" element={<StudentLayout />}>
          <Route index element={<StudentDashboard />} />
          <Route path="subjects" element={<StudentSubjects />} />
          <Route path="history" element={<StudentHistory />} />
          <Route path="profile" element={<StudentProfile />} />
        </Route>
        <Route path="*" element={<Navigate to="/student" replace />} />
      </Routes>
    );
  }

  // Teacher / Admin
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="classes" element={<Classes />} />
        <Route path="classes/:classId" element={<ClassDetail />} />
        <Route path="students" element={<Students />} />
        <Route path="face-enrollment" element={<FaceEnrollment />} />
        <Route path="face-enrollment/:studentId" element={<FaceEnrollment />} />
        <Route path="reports" element={<Reports />} />
        <Route path="settings" element={<Settings />} />
        <Route path="integrations" element={<Integrations />} />
        <Route path="history" element={<History />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
