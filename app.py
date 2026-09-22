"""
app.py — Main Streamlit application for Face Recognition Attendance System.
Role routing:
  staff   → Register Student | Take Attendance | Attendance History (read-only)
  faculty → Review & Post    | Attendance History (full)
"""
import io
import os
import json
import pickle
import sqlite3
import tempfile
from datetime import datetime, date
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace
import auth
import db
import face_utils
# ===========================================================================
# Global Constants
# ===========================================================================
MATCH_THRESHOLD = 1.04   # Official DeepFace euclidean_l2 cutoff for Facenet512
DEPARTMENTS = [
    "CSE", "CSE-IOT", "ECE", "ECE-VLSI",
    "Data Science", "Cyber Security", "Business Systems", "Civil",
]
YEARS = ["1", "2", "3", "4"]
DEFAULT_SECTIONS = ["A", "B", "C"]
DEPARTMENT_SECTIONS = {
    "CSE":              ["A", "B", "C"],
    "CSE-IOT":          ["A", "B"],
    "ECE":              ["A", "B"],
    "ECE-VLSI":         ["A", "B"],
    "Data Science":     ["A", "B"],
    "Cyber Security":   ["A", "B"],
    "Business Systems": ["A", "B"],
    "Civil":            ["A", "B"],
}
# ===========================================================================
# Page Config (must be first Streamlit call)
# ===========================================================================
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="🎓",
    layout="wide",
)
# ===========================================================================
# Database bootstrap
# ===========================================================================
db.check_database_exists()
# Ensure attendance table has the approval_status column (migration guard)
_conn = db.get_db_connection()
_cursor = _conn.cursor()
_cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id      INTEGER NOT NULL,
        date            TEXT    NOT NULL,
        time            TEXT    NOT NULL,
        status          TEXT    NOT NULL,
        confidence      REAL,
        approval_status TEXT    NOT NULL DEFAULT 'pending',
        marked_by       TEXT,
        FOREIGN KEY (student_id) REFERENCES students(id)
    )
""")
# Add approval_status column if the table existed before this upgrade
try:
    _cursor.execute("ALTER TABLE attendance ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'pending'")
except sqlite3.OperationalError:
    pass  # Column already exists
try:
    _cursor.execute("ALTER TABLE attendance ADD COLUMN marked_by TEXT")
except sqlite3.OperationalError:
    pass
_conn.commit()
_conn.close()
# Seed prototype users
auth.seed_users()
# ===========================================================================
# DeepFace model cache
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_face_model():
    return DeepFace.build_model("Facenet512")
with st.spinner("Loading face model, first run takes ~1 minute..."):
    load_face_model()
# ===========================================================================
# Helper: Embedding utilities
# ===========================================================================
def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    if embedding is None:
        return None
    embedding = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0 or np.isnan(norm):
        return embedding
    return embedding / norm
def deserialize_embeddings(raw_data) -> list:
    if raw_data is None:
        return []
    data = None
    if isinstance(raw_data, bytes):
        try:
            data = pickle.loads(raw_data)
        except Exception:
            try:
                data = json.loads(raw_data.decode("utf-8"))
            except Exception:
                return []
    elif isinstance(raw_data, str):
        try:
            data = json.loads(raw_data)
        except Exception:
            return []
    else:
        data = raw_data
    embeddings = []
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], (list, np.ndarray)):
            for item in data:
                embeddings.append(np.array(item, dtype=np.float32))
        elif isinstance(data[0], (float, int)):
            embeddings.append(np.array(data, dtype=np.float32))
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            embeddings.append(data.astype(np.float32))
        elif data.ndim == 2:
            for row in data:
                embeddings.append(row.astype(np.float32))
    return embeddings

# ===========================================================================
# Session state initialisation
# ===========================================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "role" not in st.session_state:
    st.session_state["role"] = ""
# ===========================================================================
# LOGIN PAGE
# ===========================================================================
def show_login():
    st.title("🎓 Face Recognition Attendance System")
    st.markdown("Please log in to continue.")
    col, _ = st.columns([1, 2])
    with col:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            role = auth.verify_login(username, password)
            if role:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = role
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.markdown("---")
    st.caption("**Prototype credentials:**")
    st.caption("Staff: `staff1 / staff123`  |  CR: `cr1 / cr123`  |  Faculty: `faculty1 / fac123`")


# ===========================================================================
# PAGE: Register Student  (staff only)
# ===========================================================================
def page_register():
    st.title("🎓 Student Registration")
    st.markdown("Register a new student with **3 face photos** for multi-angle biometric enrollment.")
    col_info, col_cam = st.columns([1, 1], gap="large")
    with col_info:
        st.subheader("📋 Student Details")
        name = st.text_input("Full Name *", placeholder="e.g. Jane Doe").strip()
        roll_no = st.text_input("Roll Number *", placeholder="e.g. 2024CS101").strip().upper()
        department = st.selectbox("Department *", DEPARTMENTS, index=0, key="reg_dept")
        col_y, col_s = st.columns(2)
        with col_y:
            year = st.selectbox("Year of Study *", YEARS, index=0, key="reg_year")
        with col_s:
            _reg_sections = DEPARTMENT_SECTIONS.get(department, DEFAULT_SECTIONS)
            section = st.selectbox("Section *", _reg_sections, index=0, key=f"reg_sec_{department}")
        st.info(
            "💡 **Enrollment Tips:**\n"
            "- Photo 1: Look straight, neutral expression.\n"
            "- Photo 2: Slight head turn.\n"
            "- Photo 3: Normal classroom lighting / smile."
        )

    with col_cam:
        st.subheader("📸 3-Photo Face Capture")
        has_p1 = st.session_state.get("reg_cam_1") is not None
        has_p2 = st.session_state.get("reg_cam_2") is not None
        has_p3 = st.session_state.get("reg_cam_3") is not None
        captured_count = sum([has_p1, has_p2, has_p3])
        st.progress(captured_count / 3.0, text=f"Captured {captured_count} of 3 required photos")
        tab1, tab2, tab3 = st.tabs([
            f"Photo 1 {'✅' if has_p1 else '📸'}",
            f"Photo 2 {'✅' if has_p2 else '📸'}",
            f"Photo 3 {'✅' if has_p3 else '📸'}",
        ])
        with tab1:
            photo1 = st.camera_input("Capture Photo 1 (Front)", key="reg_cam_1")
        with tab2:
            photo2 = st.camera_input("Capture Photo 2 (Slight Angle)", key="reg_cam_2")
        with tab3:
            photo3 = st.camera_input("Capture Photo 3 (Expression / Lighting)", key="reg_cam_3")
    st.markdown("---")
    register_clicked = st.button("✅ Register Student (3 Photos)", type="primary", use_container_width=True)
    if register_clicked:
        if not name:
            st.error("Please enter the student's full name.")
        elif not roll_no:
            st.error("Please enter the roll number.")
        elif not department:
            st.error("Please specify the department.")
        elif not section:
            st.error("Please enter the section.")
        elif not (photo1 and photo2 and photo3):
            missing = []
            if not photo1: missing.append("Photo 1")
            if not photo2: missing.append("Photo 2")
            if not photo3: missing.append("Photo 3")
            st.error(f"Please capture all 3 photos. Missing: {', '.join(missing)}")
        else:
            os.makedirs("faces", exist_ok=True)
            photos = [photo1, photo2, photo3]
            saved_paths, embeddings = [], []
            extraction_failed = False
            with st.spinner("Extracting 512-d embeddings for all 3 photos..."):
                for idx, photo in enumerate(photos, start=1):
                    img_path = os.path.join("faces", f"{roll_no}_{idx}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(photo.getvalue())
                    saved_paths.append(img_path)
                    try:
                        emb = face_utils.get_face_encoding(img_path)
                        embeddings.append(emb)
                    except ValueError as ve:
                        st.error(f"❌ Face detection failed for Photo {idx}. Retake in better lighting.")
                        extraction_failed = True
                        break
                    except Exception as ex:
                        st.error(f"❌ Error for Photo {idx}: {ex}")
                        extraction_failed = True
                        break
            if not extraction_failed and len(embeddings) == 3:
                embeddings_blob = pickle.dumps(embeddings)
                conn = db.get_db_connection()
                try:
                    conn.execute(
                        "INSERT INTO students (name, roll_no, department, year, section, face_embedding) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (name, roll_no, department, int(year), section, embeddings_blob),
                    )
                    conn.commit()
                    st.success(f"🎉 **{name}** (`{roll_no}`) registered successfully!")
                    p1, p2, p3 = st.columns(3)
                    with p1: st.image(saved_paths[0], caption="Photo 1", use_container_width=True)
                    with p2: st.image(saved_paths[1], caption="Photo 2", use_container_width=True)
                    with p3: st.image(saved_paths[2], caption="Photo 3", use_container_width=True)
                except sqlite3.IntegrityError:
                    st.error(f"⚠️ Roll Number **'{roll_no}'** already exists in the database.")
                finally:
                    conn.close()



# ===========================================================================
# PAGE: Take Attendance  (staff only)
# ===========================================================================
def page_take_attendance():
    st.title("📝 Live Face Attendance")
    st.markdown("Select the class, snap a live photo — attendance is saved as **Pending** for faculty review.")
    username = st.session_state["username"]
    st.subheader("🏫 Class Selection")
    c1, c2, c3 = st.columns(3)
    with c1:
        dept_filter = st.selectbox("Department", DEPARTMENTS, key="att_dept")
    with c2:
        year_filter = st.selectbox("Year", YEARS, key="att_year")
    with c3:
        _att_sections = DEPARTMENT_SECTIONS.get(dept_filter, DEFAULT_SECTIONS)
        sec_filter = st.selectbox("Section", _att_sections, key=f"att_sec_{dept_filter}")
    conn = db.get_db_connection()
    enrolled_students = conn.execute(
        "SELECT id, name, roll_no, face_embedding FROM students "
        "WHERE department=? AND year=? AND section=? AND face_embedding IS NOT NULL",
        (dept_filter, int(year_filter), sec_filter),
    ).fetchall()
    conn.close()
    total = len(enrolled_students)
    if total == 0:
        st.warning(f"No enrolled students found for {dept_filter} / Year {year_filter} / Section {sec_filter}.")
    else:
        st.info(f"👥 **{total}** enrolled student(s) in this class.")
    with st.expander("⚙️ Advanced Recognition Settings", expanded=False):
        threshold = st.slider(
            "Normalized Distance Threshold (Facenet512 default: 1.04)",
            0.30, 1.40, MATCH_THRESHOLD, 0.01,
        )
    st.markdown("---")
    col_cam, col_result = st.columns([1, 1], gap="large")
    with col_cam:
        st.subheader("📸 Live Student Face")
        att_photo = st.camera_input("Capture student face", key="att_camera")
        mark_clicked = st.button("📝 Mark Attendance", type="primary", use_container_width=True)
    with col_result:
        st.subheader("📊 Recognition Result")
        r_ph = st.empty()
        m_ph = st.empty()
        d_ph = st.empty()
        if mark_clicked:
            if not dept_filter or not sec_filter:
                r_ph.error("Please select a valid Department and Section.")
            elif total == 0:
                r_ph.error("No enrolled students in this class.")
            elif att_photo is None:
                r_ph.error("Please take a photo first.")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(att_photo.getvalue())
                tmp.close()
                try:
                    with st.spinner("Detecting faces..."):
                        num_faces = face_utils.count_faces(tmp.name)

                    if num_faces == 0:
                        r_ph.error("❌ No face detected. Retake in better lighting.")
                    elif num_faces > 1:
                        r_ph.warning("⚠️ Only ONE person at a time in front of the camera.")
                    else:
                        with st.spinner("Extracting live face embedding..."):
                            live_raw = face_utils.get_face_encoding(tmp.name)
                            live_norm = l2_normalize(live_raw)
                        best_match = None
                        best_norm_d = float("inf")
                        best_raw_d = float("inf")
                        debug_rows = []
                        for student in enrolled_students:
                            stored_list = deserialize_embeddings(student["face_embedding"])
                            if not stored_list:
                                continue
                            s_norm_d = float("inf")
                            s_raw_d = float("inf")
                            for stored_raw in stored_list:
                                s_norm = l2_normalize(stored_raw)
                                raw_d = float(np.linalg.norm(live_raw - stored_raw))
                                _, norm_d = face_utils.compare_faces(live_norm, s_norm, threshold=threshold)
                                norm_d = float(norm_d)
                                if norm_d < s_norm_d:
                                    s_norm_d, s_raw_d = norm_d, raw_d
                            is_match = s_norm_d < threshold
                            debug_rows.append({
                                "Roll No": student["roll_no"],
                                "Name": student["name"],
                                "Photos": len(stored_list),
                                "Min Raw Dist": round(s_raw_d, 4),
                                "Min Norm Dist": round(s_norm_d, 4),
                                "Threshold": round(threshold, 2),
                                "Status": "✅ Matched" if is_match else "❌ Exceeded",
                            })
                            if s_norm_d < best_norm_d:
                                best_norm_d, best_raw_d, best_match = s_norm_d, s_raw_d, student
                        debug_rows.sort(key=lambda x: x["Min Norm Dist"])
                        with d_ph.container():
                            with st.expander("🛠️ Candidate Breakdown", expanded=True):
                                st.caption(f"Threshold: `{threshold}`")
                                st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
                        if best_match and best_norm_d < threshold:
                            sid = best_match["id"]
                            sname = best_match["name"]
                            sroll = best_match["roll_no"]
                            today = datetime.now().strftime("%Y-%m-%d")
                            now_time = datetime.now().strftime("%H:%M:%S")
                            cos_sim = 1 - (best_norm_d ** 2) / 2
                            conf = round(cos_sim * 100, 1)
                            conn = db.get_db_connection()
                            # Duplicate check: any row (pending OR posted) same student + date
                            existing = conn.execute(
                                "SELECT time, approval_status FROM attendance "
                                "WHERE student_id=? AND date=?",
                                (sid, today),
                            ).fetchone()
                            if existing:
                                r_ph.info(
                                    f"ℹ️ Attendance already exists for **{sname}** (`{sroll}`) "
                                    f"on {today} at {existing['time']} "
                                    f"[{existing['approval_status']}]."
                                )
                            else:
                                conn.execute(
                                    "INSERT INTO attendance "
                                    "(student_id, date, time, status, confidence, approval_status, marked_by) "
                                    "VALUES (?, ?, ?, 'Present', ?, 'pending', ?)",
                                    (sid, today, now_time, conf, username),
                                )
                                conn.commit()
                                r_ph.success(
                                    f"✅ **{sname}** (`{sroll}`) marked **Present** — "
                                    f"saved as **Pending** (confidence: {conf}%)"
                                )
                            conn.close()
                            with m_ph.container():
                                mc1, mc2 = st.columns(2)
                                mc1.metric("Recognized", f"{sname} ({sroll})")
                                mc2.metric("Confidence", f"{conf}%")
                                st.caption(
                                    f"📏 Norm Dist: `{best_norm_d:.4f}` | "
                                    f"Raw Dist: `{best_raw_d:.4f}` | "
                                    f"Threshold: `{threshold}`"
                                )
                        else:
                            r_ph.warning("⚠️ Face not recognized or not enrolled in this class.")
                            with m_ph.container():
                                if best_match:
                                    st.caption(
                                        f"Closest: **{best_match['name']}** "
                                        f"norm dist `{best_norm_d:.4f}` > threshold `{threshold}`"
                                    )
                except ValueError:
                    r_ph.error("❌ No face detected. Retake with good lighting.")
                except Exception as ex:
                    r_ph.error(f"❌ Error: {ex}")
                finally:
                    if os.path.exists(tmp.name):
                        try:
                            os.remove(tmp.name)
                        except OSError:
                            pass


# ===========================================================================
# PAGE: Review & Post  (faculty only)
# ===========================================================================
def page_review_post():
    st.title("📋 Review & Post Attendance")
    st.markdown("Review pending records, add/remove entries, then post to finalise.")
    # ---- Filters ----
    st.subheader("🔍 Select Class & Date")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        rp_dept = st.selectbox("Department", DEPARTMENTS, key="rp_dept")
    with fc2:
        rp_year = st.selectbox("Year", YEARS, key="rp_year")
    with fc3:
        _rp_sections = DEPARTMENT_SECTIONS.get(rp_dept, DEFAULT_SECTIONS)
        rp_sec = st.selectbox("Section", _rp_sections, key=f"rp_sec_{rp_dept}")
    with fc4:
        rp_date = st.date_input("Date", value=date.today(), key="rp_date")
    rp_date_str = rp_date.strftime("%Y-%m-%d")
    st.markdown("---")
    conn = db.get_db_connection()
    # Fetch all students in selected class
    all_students = conn.execute(
        "SELECT id, name, roll_no FROM students "
        "WHERE department=? AND year=? AND section=?",
        (rp_dept, int(rp_year), rp_sec),
    ).fetchall()
    all_students_map = {s["id"]: s for s in all_students}
   # Fetch pending attendance for selected class + date
    pending_rows = conn.execute(
        """
        SELECT a.id, a.student_id, s.name, s.roll_no, a.status, a.confidence,
               a.time, a.marked_by, a.approval_status
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE s.department=? AND s.year=? AND s.section=? AND a.date=?
          AND a.approval_status = 'pending'
        ORDER BY s.roll_no
        """,
        (rp_dept, int(rp_year), rp_sec, rp_date_str),
    ).fetchall()
    conn.close()
    if not all_students:
        st.warning(f"No students enrolled in {rp_dept} / Year {rp_year} / Section {rp_sec}.")
        return
    st.subheader(f"📄 Pending Records — {rp_dept}, Year {rp_year}, Section {rp_sec}, {rp_date_str}")
    if not pending_rows:
        st.info("No pending attendance records for this class and date.")
    else:
        st.caption(f"{len(pending_rows)} pending record(s). Edit below, then click **Post Attendance**.")
    # ---- Editable pending table via session state ----
    ss_key = f"rp_records_{rp_dept}_{rp_year}_{rp_sec}_{rp_date_str}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = [
            {
                "att_id": r["id"],
                "student_id": r["student_id"],
                "name": r["name"],
                "roll_no": r["roll_no"],
                "status": r["status"],
                "confidence": r["confidence"],
                "time": r["time"],
                "marked_by": r["marked_by"],
                "to_remove": False,
            }
            for r in pending_rows
        ]
    records = st.session_state[ss_key]
    # Display editable rows
    header_cols = st.columns([2, 3, 2, 2, 2, 1])
    header_cols[0].markdown("**Roll No**")
    header_cols[1].markdown("**Name**")
    header_cols[2].markdown("**Status**")
    header_cols[3].markdown("**Confidence**")
    header_cols[4].markdown("**Marked By**")
    header_cols[5].markdown("**Remove**")
    for i, rec in enumerate(records):
        if rec["to_remove"]:
            continue
        c0, c1, c2, c3, c4, c5 = st.columns([2, 3, 2, 2, 2, 1])
        c0.write(rec["roll_no"])
        c1.write(rec["name"])
        new_status = c2.selectbox(
            "status",
            ["Present", "Absent"],
            index=0 if rec["status"] == "Present" else 1,
            key=f"status_{ss_key}_{i}",
            label_visibility="collapsed",
        )
        records[i]["status"] = new_status
        c3.write(f"{rec['confidence']:.1f}%" if rec["confidence"] else "—")
        c4.write(rec.get("marked_by") or "—")
        if c5.button("🗑️", key=f"remove_{ss_key}_{i}", help="Remove this entry"):
            records[i]["to_remove"] = True
            st.rerun()
    # ---- Add a student manually ----
    st.markdown("---")
    with st.expander("➕ Add Student Manually (latecomers, absences, flagged faces)"):
        # Students in class who don't already have a record (pending or posted) for this date
        conn2 = db.get_db_connection()
        already_ids = {r["student_id"] for r in pending_rows}
        # Also check posted records for same date
        posted_ids_today = {
            row["student_id"] for row in conn2.execute(
                "SELECT student_id FROM attendance "
                "WHERE date=? AND approval_status='posted'",
                (rp_date_str,),
            ).fetchall()
        }
        conn2.close()
        all_existing_ids = already_ids | posted_ids_today
        pending_ids_in_session = {r["student_id"] for r in records if not r["to_remove"]}
        available = [
            s for s in all_students
            if s["id"] not in all_existing_ids and s["id"] not in pending_ids_in_session
        ]
        if not available:
            st.info("All enrolled students already have a record for this date.")
        else:
            add_options = {f"{s['roll_no']} — {s['name']}": s for s in available}
            add_choice = st.selectbox("Select Student", list(add_options.keys()), key=f"add_student_{ss_key}")
            add_status = st.selectbox("Status to assign", ["Present", "Absent"], key=f"add_status_{ss_key}")
            if st.button("Add to List", key=f"add_btn_{ss_key}"):
                chosen = add_options[add_choice]
                records.append({
                    "att_id": None,          # New record — no DB id yet
                    "student_id": chosen["id"],
                    "name": chosen["name"],
                    "roll_no": chosen["roll_no"],
                    "status": add_status,
                    "confidence": None,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "marked_by": st.session_state["username"],
                    "to_remove": False,
                })
                st.rerun()
    # ---- Post Attendance button with confirmation ----
    st.markdown("---")
    active_records = [r for r in records if not r["to_remove"]]
    st.info(f"**{len(active_records)}** record(s) will be posted upon confirmation.")
    if "rp_confirm_pending" not in st.session_state:
        st.session_state["rp_confirm_pending"] = False
    col_post, col_cancel = st.columns([1, 4])
    with col_post:
        if st.button("📬 Post Attendance", type="primary", use_container_width=True):
            st.session_state["rp_confirm_pending"] = True
    if st.session_state["rp_confirm_pending"]:
        st.warning(
            f"⚠️ You are about to **post {len(active_records)} record(s)** for "
            f"{rp_dept} / Year {rp_year} / Section {rp_sec} on **{rp_date_str}**. "
            "This cannot be undone."
        )
        cc1, cc2, _ = st.columns([1, 1, 3])
        with cc1:
            if st.button("✅ Confirm Post", type="primary"):
                conn3 = db.get_db_connection()
                present_count = 0
                absent_count = 0
                for rec in active_records:
                    if rec["att_id"] is not None:
                        # Update existing pending row
                        conn3.execute(
                            "UPDATE attendance SET status=?, approval_status='posted' WHERE id=?",
                            (rec["status"], rec["att_id"]),
                        )
                    else:
                        # Insert new row as posted directly (faculty-added)
                        conn3.execute(
                            "INSERT INTO attendance "
                            "(student_id, date, time, status, confidence, approval_status, marked_by) "
                            "VALUES (?, ?, ?, ?, ?, 'posted', ?)",
                            (
                                rec["student_id"], rp_date_str,
                                rec["time"], rec["status"],
                                rec["confidence"],
                                rec["marked_by"],
                            ),
                        )
                    if rec["status"] == "Present":
                        present_count += 1
                    else:
                        absent_count += 1
                # Delete removed pending rows from DB
                for rec in records:
                    if rec["to_remove"] and rec["att_id"] is not None:
                        conn3.execute("DELETE FROM attendance WHERE id=?", (rec["att_id"],))
                conn3.commit()
                conn3.close()
                # Clear session state for this filter
                del st.session_state[ss_key]
                st.session_state["rp_confirm_pending"] = False
                st.success(
                    f"✅ Posted **{len(active_records)}** record(s) — "
                    f"**{present_count} Present**, **{absent_count} Absent**."
                )
                st.rerun()
        with cc2:
            if st.button("❌ Cancel"):
                st.session_state["rp_confirm_pending"] = False
                st.rerun()
# ===========================================================================
# PAGE: Attendance History  (both roles)
# ===========================================================================
def page_history(role: str):
    st.title("📊 Attendance History")
    is_faculty = role == "faculty"
    # ---- Filters ----
    st.subheader("🔍 Filters")
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        h_dept = st.selectbox("Department", ["All"] + DEPARTMENTS, key="hist_dept")
    with hc2:
        h_year = st.selectbox("Year", ["All"] + YEARS, key="hist_year")
    with hc3:
        _hist_sections = (
            DEFAULT_SECTIONS if h_dept == "All"
            else DEPARTMENT_SECTIONS.get(h_dept, DEFAULT_SECTIONS)
        )
        h_sec = st.selectbox("Section", ["All"] + _hist_sections, key=f"hist_sec_{h_dept}")
    hd1, hd2 = st.columns(2)
    with hd1:
        h_from = st.date_input("From Date", value=date.today(), key="hist_from")
    with hd2:
        h_to = st.date_input("To Date", value=date.today(), key="hist_to")
    show_pending = False
    if is_faculty:
        show_pending = st.checkbox("Also show Pending records", value=False, key="hist_pending")
    else:
        st.caption("Showing **Posted** records. Pending records visible to Faculty only.")
    # Build query
    where_parts = ["a.date >= ?", "a.date <= ?"]
    params = [h_from.strftime("%Y-%m-%d"), h_to.strftime("%Y-%m-%d")]
    if h_dept != "All":
        where_parts.append("s.department = ?")
        params.append(h_dept)
    if h_year != "All":
        where_parts.append("s.year = ?")
        params.append(int(h_year))
    if h_sec != "All":
        where_parts.append("s.section = ?")
        params.append(h_sec)
    if show_pending:
        where_parts.append("a.approval_status IN ('pending', 'posted')")
    else:
        where_parts.append("a.approval_status = 'posted'")
    where_clause = " AND ".join(where_parts)
    query = f"""
        SELECT s.roll_no, s.name, s.department, s.year, s.section,
               a.date, a.time, a.status, a.confidence,
               a.approval_status, a.marked_by
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE {where_clause}
        ORDER BY a.date DESC, s.roll_no
    """
    conn = db.get_db_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    df = pd.DataFrame(
        [dict(r) for r in rows],
        columns=["roll_no", "name", "department", "year", "section",
                 "date", "time", "status", "confidence", "approval_status", "marked_by"],
    )
    if df.empty:
        st.info("No attendance records match the selected filters.")
        return
    # Rename columns for display
    df_display = df.rename(columns={
        "roll_no": "Roll No", "name": "Name", "department": "Dept",
        "year": "Year", "section": "Sec", "date": "Date", "time": "Time",
        "status": "Status", "confidence": "Confidence (%)",
        "approval_status": "Approval", "marked_by": "Marked By",
    })
    if "Confidence (%)" in df_display.columns:
        df_display["Confidence (%)"] = df_display["Confidence (%)"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )
    st.dataframe(df_display, use_container_width=True)
    st.caption(f"Showing **{len(df)}** record(s).")
    # ---- CSV Export (posted only) ----
    posted_df = df[df["approval_status"] == "posted"].drop(columns=["approval_status"])
    if posted_df.empty:
        st.caption("No posted records to export for the current filter.")
    else:
        csv_buf = io.StringIO()
        posted_df.rename(columns={
            "roll_no": "Roll No", "name": "Name", "department": "Department",
            "year": "Year", "section": "Section", "date": "Date (YYYY-MM-DD)",
            "time": "Time", "status": "Status", "confidence": "Confidence (%)",
            "marked_by": "Marked By",
        }).to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Export Posted Records as CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"attendance_posted_{h_from}_{h_to}.csv",
            mime="text/csv",
        )
# ===========================================================================
# MAIN ROUTING
# ===========================================================================
if not st.session_state["logged_in"]:
    show_login()
else:
    role = st.session_state["role"]
    username = st.session_state["username"]
    # ---- Sidebar ----
    with st.sidebar:
        st.title("🎓 Attendance System")
        st.caption(f"Logged in as **{username}** ({role.title()})")
        st.markdown("---")
        if role == "staff":
            page = st.radio(
                "Navigation",
                ["Register Student", "Take Attendance", "Attendance History"],
            )
        else:  # faculty
            page = st.radio(
                "Navigation",
                ["Review & Post", "Attendance History"],
            )
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["logged_in", "username", "role"]:
                st.session_state[key] = "" if key != "logged_in" else False
            st.rerun()
    # ---- Page dispatch ----
    if page == "Register Student":
        page_register()
    elif page == "Take Attendance":
        page_take_attendance()
    elif page == "Review & Post":
        page_review_post()
    elif page == "Attendance History":
        page_history(role)
# To run type .\venv\Scripts\streamlit.exe run app.py
