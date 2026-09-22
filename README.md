# 🎓 Smart Campus Attendance System using ML-Driven Face Recognition

A role-based, biometric attendance management system built with **Streamlit** and **DeepFace (Facenet512)**. Staff capture live student faces to mark attendance; faculty review, edit, and post the records. No RFID cards, no manual registers.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🔐 **Role-based login** | STAFF (mark attendance) · FACULTY (review & post) |
| 📸 **3-photo enrollment** | Multi-angle biometric registration per student |
| 🧠 **ML face recognition** | DeepFace Facenet512 with L2-normalized Euclidean matching |
| 👁️ **Single-person guard** | Rejects photos with 0 or more than 1 face |
| ✅ **Approval workflow** | Attendance saved as *Pending* → Faculty posts as *Posted* |
| 📋 **Review & Post page** | Faculty can add/remove/edit entries before finalising |
| 📊 **Attendance History** | Filter by department, year, section, date range |
| ⬇️ **CSV Export** | One-click export of posted attendance records |
| 🔄 **Cascading dropdowns** | Section options auto-update based on selected department |

---

## 🏗️ Tech Stack

- **Frontend / UI** — [Streamlit](https://streamlit.io/)
- **Face Recognition** — [DeepFace](https://github.com/serengil/deepface) · Facenet512 model · RetinaFace detector
- **Database** — SQLite (via Python `sqlite3`)
- **Image Processing** — OpenCV · NumPy
- **Auth** — SHA-256 hashed passwords (prototype)

---

## 📁 Project Structure

```
Face_Attendance_System/
│
├── app.py               # Main Streamlit application (all pages & routing)
├── auth.py              # Login / role verification helpers
├── db.py                # SQLite connection & table initialisation
├── face_utils.py        # get_face_encoding(), compare_faces(), count_faces()
├── reset_db.py          # Utility to wipe attendance/users (preserves students)
├── requirements.txt     # Python dependencies
├── haarcascade_frontalface_default.xml
│
├── faces/               # Enrolled face photos (gitignored — biometric data)
└── attendance.db        # SQLite database (gitignored — personal data)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ritwikvins20/Smart_Campus_Attendance_System_using_ML-Driven_Face_Recognition.git
cd Smart_Campus_Attendance_System_using_ML-Driven_Face_Recognition
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the Facenet512 model weights (~250 MB). This happens automatically.

### 4. Run the app
```bash
streamlit run app.py
# or on Windows with the venv directly:
.\venv\Scripts\streamlit.exe run app.py
```

---

## 👤 Prototype Login Credentials

| Username | Password | Role |
|----------|----------|------|
| `staff1` | `staff123` | Staff (mark attendance) |
| `cr1` | `cr123` | Staff (mark attendance) |
| `faculty1` | `fac123` | Faculty (review & post) |

> ⚠️ Passwords use SHA-256 (no salt) — prototype only. Replace with bcrypt/argon2 for production.

---

## 🗄️ Database Schema

```sql
students (id, name, roll_no, department, year, section, face_embedding BLOB)

attendance (id, student_id, date, time, status,
            confidence, approval_status, marked_by)

users (id, username, password_hash, role)
```

- `face_embedding` stores a pickled list of 3 Facenet512 embeddings (one per enrollment photo).
- `approval_status` is `'pending'` when marked by staff, `'posted'` after faculty approval.

---

## 🔄 Attendance Workflow

```
Staff logs in
    └─▶ Selects Department / Year / Section
    └─▶ Captures live student face
    └─▶ System matches against enrolled embeddings (min-distance across 3 photos)
    └─▶ Attendance saved as PENDING

Faculty logs in
    └─▶ Opens "Review & Post" page
    └─▶ Reviews pending records (can add / remove / change status)
    └─▶ Clicks "Post Attendance" (2-click confirmation)
    └─▶ Records flipped to POSTED → visible in History & CSV export
```

---

## 🧪 Resetting the Database

```bash
# Wipe attendance + users (keep registered students & embeddings)
python reset_db.py

# Wipe everything including students
python -c "
import sqlite3; conn = sqlite3.connect('attendance.db')
conn.executescript('DELETE FROM students; DELETE FROM attendance; DELETE FROM users;')
conn.commit(); conn.close(); print('Done')
"

# Also delete saved face photos
rm -rf faces/   # Linux/macOS
Remove-Item -Recurse -Force .\faces\   # Windows PowerShell
```

---

## 📌 Configuration

All class options are defined as constants at the **top of `app.py`** — edit them freely:

```python
DEPARTMENTS = ["CSE", "CSE-IOT", "ECE", "ECE-VLSI", "Data Science", "Cyber Security", "Business Systems", "Civil"]
YEARS = ["1", "2", "3", "4"]
DEFAULT_SECTIONS = ["A", "B", "C"]
DEPARTMENT_SECTIONS = {
    "CSE": ["A", "B", "C"],
    "CSE-IOT": ["A", "B"],
    # ...
}
MATCH_THRESHOLD = 1.04   # euclidean_l2 cutoff for Facenet512
```

---

## 📄 License

This project is for academic and educational purposes.
