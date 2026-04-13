import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

// ── JWT Interceptor ──────────────────────────────────────────────────────────
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.href = "/";
    }
    return Promise.reject(err);
  }
);

// ── Auth ─────────────────────────────────────────────────────────────────────
export const authAPI = {
  loginTeacher:  (body) => api.post("/auth/login", body),
  loginStudent:  (body) => api.post("/auth/student-login", body),
  loginCollege:  (body) => api.post("/auth/college-token", body),
  getMe:         ()     => api.get("/auth/me"),
};

// ── Classes ──────────────────────────────────────────────────────────────────
export const classesAPI = {
  list:           ()         => api.get("/classes"),
  create:         (body)     => api.post("/classes", body),
  get:            (id)       => api.get(`/classes/${id}`),
  update:         (id, body) => api.put(`/classes/${id}`, body),
  delete:         (id)       => api.delete(`/classes/${id}`),
  listStudents:   (id)       => api.get(`/classes/${id}/students`),
  enrollStudents: (id, body) => api.post(`/classes/${id}/students`, body),
  removeStudent:  (cid, sid) => api.delete(`/classes/${cid}/students/${sid}`),
};

// ── Students ─────────────────────────────────────────────────────────────────
export const studentsAPI = {
  list:         (params)     => api.get("/students", { params }),
  create:       (body)       => api.post("/students", body),
  get:          (id)         => api.get(`/students/${id}`),
  update:       (id, body)   => api.put(`/students/${id}`, body),
  delete:       (id)         => api.delete(`/students/${id}`),
  bulkImport:   (formData)   => api.post("/students/bulk-import", formData, { headers: { "Content-Type": "multipart/form-data" } }),
  attendance:   (id, params) => api.get(`/students/${id}/attendance`, { params }),
  stats:        (id)         => api.get(`/attendance/student/${id}/stats`),
};

// ── Face Enrollment ───────────────────────────────────────────────────────────
export const faceAPI = {
  enroll:       (formData) => api.post("/register", formData, { headers: { "Content-Type": "multipart/form-data" } }),
  clearFace:    (studentId) => api.delete(`/register/${studentId}`),
  getQuality:   (studentId) => api.get(`/register/${studentId}/quality`),
};

// ── Attendance ────────────────────────────────────────────────────────────────
export const attendanceAPI = {
  list:          (params)     => api.get("/attendance", { params }),
  override:      (id, body)   => api.put(`/attendance/${id}`, body),
  summary:       (params)     => api.get("/attendance/reports/summary", { params }),
  defaulters:    (class_id)   => api.get("/attendance/reports/defaulters", { params: { class_id } }),
  studentStats:  (studentId)  => api.get(`/attendance/student/${studentId}/stats`),
  recognize:     (formData, params) => api.post("/recognize", formData, {
    params,
    headers: { "Content-Type": "multipart/form-data" },
  }),
};

// ── Integration ───────────────────────────────────────────────────────────────
export const integrateAPI = {
  listColleges:    ()          => api.get("/integrate/colleges"),
  createCollege:   (body)      => api.post("/integrate/colleges", body),
  regenerateKey:   (id)        => api.post(`/integrate/colleges/${id}/regenerate-key`),
  getStudents:     ()          => api.get("/integrate/students"),
  enroll:          (formData)  => api.post("/integrate/enroll", formData, { headers: { "Content-Type": "multipart/form-data" } }),
  logAttendance:   (body)      => api.post("/integrate/attendance", body),
  getAttendance:   (rollNo)    => api.get(`/integrate/attendance/${rollNo}`),
};

export default api;
