import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";
import { Toaster } from "react-hot-toast";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
    <Toaster
      position="top-right"
      toastOptions={{
        style: {
          background: "var(--bg-card)",
          color: "var(--text)",
          border: "1px solid var(--border)",
          borderRadius: "10px",
          fontSize: "0.875rem",
          fontFamily: "Inter, sans-serif",
          boxShadow: "var(--shadow-lg)",
        },
        success: { iconTheme: { primary: "#10b981", secondary: "#fff" } },
        error:   { iconTheme: { primary: "#ef4444", secondary: "#fff" } },
      }}
    />
  </React.StrictMode>
);
