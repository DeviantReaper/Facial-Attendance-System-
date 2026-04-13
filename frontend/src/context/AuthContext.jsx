import React, { createContext, useContext, useState, useCallback } from "react";
import { authAPI } from "../services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const stored = () => {
    try {
      const u = localStorage.getItem("user");
      const t = localStorage.getItem("token");
      return u && t ? JSON.parse(u) : null;
    } catch { return null; }
  };

  const [auth, setAuth] = useState(stored);

  const login = useCallback(async (username, password, role = "teacher") => {
    let res;
    if (role === "student") {
      res = await authAPI.loginStudent({ roll_no: username, password });
    } else {
      res = await authAPI.loginTeacher({ username, password });
    }
    const { access_token, role: userRole, user_id, name } = res.data;
    const userData = { token: access_token, role: userRole, id: user_id, name };
    localStorage.setItem("token", access_token);
    localStorage.setItem("user", JSON.stringify(userData));
    setAuth(userData);
    return userData;
  }, []);

  // Demo/mock login for development (no backend needed)
  const mockLogin = useCallback((role, user) => {
    const userData = { role, name: user.name || user.roll_no, id: user.id || 1, token: "mock" };
    localStorage.setItem("user", JSON.stringify(userData));
    setAuth(userData);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setAuth(null);
  }, []);

  return (
    <AuthContext.Provider value={{ auth, login, mockLogin, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
