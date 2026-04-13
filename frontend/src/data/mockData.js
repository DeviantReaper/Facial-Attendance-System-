// ── Mock data shared across pages ──────────────────────────────────────────

export const mockClasses = [
  {
    id: "cs101",
    code: "CS101",
    name: "Introduction to Artificial Intelligence",
    room: "Room 204-A",
    time: "Mon / Wed  09:00 – 10:30",
    enrolled: 48,
    present: 44,
    color: "#6366f1",
  },
  {
    id: "cs202",
    code: "CS202",
    name: "Data Structures & Algorithms",
    room: "Room 110-B",
    time: "Tue / Thu  11:00 – 12:30",
    enrolled: 52,
    present: 41,
    color: "#8b5cf6",
  },
  {
    id: "cs315",
    code: "CS315",
    name: "Computer Vision",
    room: "Lab 3",
    time: "Fri  14:00 – 17:00",
    enrolled: 30,
    present: 28,
    color: "#ec4899",
  },
  {
    id: "cs420",
    code: "CS420",
    name: "Machine Learning",
    room: "Room 308",
    time: "Mon / Thu  15:00 – 16:30",
    enrolled: 42,
    present: 35,
    color: "#14b8a6",
  },
];

const firstNames = ["Aarav","Priya","Rohan","Neha","Vikram","Ananya","Kabir","Shruti","Arjun","Divya","Siddharth","Pooja","Karthik","Meera","Rahul"];
const lastNames  = ["Sharma","Patel","Singh","Kumar","Verma","Nair","Iyer","Mehta","Joshi","Gupta","Reddy","Chopra","Das","Shah","Malhotra"];

function seededInt(seed, max) {
  let s = Math.sin(seed * 9301 + 49297) * 233280;
  return Math.floor((s - Math.floor(s)) * max);
}

export function getStudentsForClass(classId) {
  const cls = mockClasses.find(c => c.id === classId);
  if (!cls) return [];
  return Array.from({ length: cls.enrolled }, (_, i) => {
    const fn = firstNames[seededInt(i * 3 + 1, firstNames.length)];
    const ln = lastNames[seededInt(i * 3 + 2, lastNames.length)];
    return {
      id: `${classId}-${String(i + 1).padStart(3, "0")}`,
      rollNo: `MUJ2022${String(i + 101).padStart(3, "0")}`,
      name: `${fn} ${ln}`,
      status: i < cls.present ? "present" : "absent",
      avatar: fn[0] + ln[0],
    };
  });
}

export const mockHistory = [
  { date: "2026-04-02", classCode: "CS101", subject: "Neural Networks – Lecture 12", present: 44, absent: 4, total: 48 },
  { date: "2026-04-02", classCode: "CS202", subject: "Graph Algorithms",            present: 41, absent: 11, total: 52 },
  { date: "2026-04-01", classCode: "CS315", subject: "Object Detection – YOLO",     present: 28, absent: 2,  total: 30 },
  { date: "2026-04-01", classCode: "CS420", subject: "Loss Functions & Optimisers",  present: 35, absent: 7,  total: 42 },
  { date: "2026-03-31", classCode: "CS101", subject: "Backpropagation",              present: 46, absent: 2,  total: 48 },
  { date: "2026-03-31", classCode: "CS202", subject: "Dynamic Programming",          present: 38, absent: 14, total: 52 },
  { date: "2026-03-30", classCode: "CS315", subject: "CNNs & Feature Maps",          present: 29, absent: 1,  total: 30 },
  { date: "2026-03-29", classCode: "CS420", subject: "Decision Trees & RF",          present: 40, absent: 2,  total: 42 },
];

// ── Student mock profiles ───────────────────────────────────────────────────

export const mockStudents = [
  { id: "s01", name: "Aarav Sharma",   rollNo: "MUJ2022101", avatar: "AS" },
  { id: "s02", name: "Priya Patel",    rollNo: "MUJ2022102", avatar: "PP" },
  { id: "s03", name: "Rohan Singh",    rollNo: "MUJ2022103", avatar: "RS" },
  { id: "s04", name: "Neha Kumar",     rollNo: "MUJ2022104", avatar: "NK" },
  { id: "s05", name: "Vikram Verma",   rollNo: "MUJ2022105", avatar: "VV" },
];

// Per-student per-subject attendance
const studentAttendanceTable = {
  s01: { cs101: { attended: 22, total: 24 }, cs202: { attended: 19, total: 24 }, cs315: { attended: 10, total: 12 }, cs420: { attended: 16, total: 20 } },
  s02: { cs101: { attended: 20, total: 24 }, cs202: { attended: 15, total: 24 }, cs315: { attended:  9, total: 12 }, cs420: { attended: 12, total: 20 } },
  s03: { cs101: { attended: 24, total: 24 }, cs202: { attended: 22, total: 24 }, cs315: { attended: 11, total: 12 }, cs420: { attended: 18, total: 20 } },
  s04: { cs101: { attended: 16, total: 24 }, cs202: { attended: 13, total: 24 }, cs315: { attended:  7, total: 12 }, cs420: { attended: 10, total: 20 } },
  s05: { cs101: { attended: 18, total: 24 }, cs202: { attended: 20, total: 24 }, cs315: { attended: 12, total: 12 }, cs420: { attended: 15, total: 20 } },
};

export function getStudentSubjects(studentId) {
  const records = studentAttendanceTable[studentId] || {};
  return mockClasses.map(cls => {
    const key = cls.id.replace("-", "");
    const rec  = records[key] || { attended: 0, total: 1 };
    const pct  = Math.round((rec.attended / rec.total) * 100);
    return { ...cls, attended: rec.attended, totalClasses: rec.total, pct };
  });
}

// Per-student session history
export function getStudentHistory(studentId) {
  const row = mockHistory.map((h, i) => ({
    ...h,
    myStatus: (i + parseInt(studentId.replace("s",""), 10)) % 5 === 0 ? "absent" : "present",
  }));
  return row;
}

// Register new mock student
export function registerMockStudent(name, rollNo) {
  const newId = `s${mockStudents.length + 1}`;
  const getInitials = (n) => {
    const parts = n.trim().split(" ");
    return parts.length > 1 ? (parts[0][0] + parts[1][0]).toUpperCase() : n.substring(0,2).toUpperCase();
  };
  const newStudent = {
    id: newId,
    name,
    rollNo: rollNo.toUpperCase(),
    avatar: getInitials(name)
  };
  mockStudents.push(newStudent);
  return newStudent;
}


