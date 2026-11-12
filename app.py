"""
app.py — Main Streamlit application for Face Recognition Attendance System.
Role routing:
  staff   → Register Student | Take Attendance | Attendance History (read-only)
  faculty → Review & Post    | Attendance History (full)
"""
import base64
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
# Design System — CampusAttend CSS (shadcn/ui-inspired clinical blueprint)
# Primary #2563EB · Canvas #F5F5F5 · Card #FFFFFF · Font: Geist/Inter
# ===========================================================================
def _inject_css():
    """Inject the CampusAttend design system CSS. Called once at startup."""
    st.markdown("""
<style>
/* ── Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #0A0A0A !important;
}

/* ── Page Background ──────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] > .main {
    background-color: #F5F5F5 !important;
}
.main .block-container {
    padding: 2rem 2.5rem 2rem 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden; display: none; }

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E5E5 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem 1rem 1rem !important;
}
/* Sidebar nav buttons: transparent, left-aligned, pill */
[data-testid="stSidebar"] [data-testid="stButton"] > button,
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background-color: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    width: 100% !important;
    height: auto !important;
    min-height: 36px !important;
    transition: background-color 0.15s ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover,
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background-color: #F5F5F5 !important;
    color: #0A0A0A !important;
}

/* ── Buttons — Primary ────────────────────────────────────────────── */
[data-testid="baseButton-primary"],
[data-testid="stButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0 1.25rem !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    height: 40px !important;
    min-height: 40px !important;
    box-shadow: 0 1px 2px rgba(37,99,235,0.25) !important;
    transition: background-color 0.15s ease !important;
}
[data-testid="baseButton-primary"]:hover,
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
}

/* ── Buttons — Secondary (main content only) ─────────────────────── */
.main [data-testid="baseButton-secondary"],
.main [data-testid="stButton"] > button:not([kind="primary"]) {
    background-color: #FFFFFF !important;
    color: #374151 !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 10px !important;
    padding: 0 1.25rem !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    height: 40px !important;
    min-height: 40px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    transition: all 0.15s ease !important;
}
.main [data-testid="baseButton-secondary"]:hover,
.main [data-testid="stButton"] > button:not([kind="primary"]):hover {
    background-color: #F9FAFB !important;
    border-color: #D1D5DB !important;
}

/* ── Text Input ───────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background-color: #F9FAFB !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.875rem !important;
    font-size: 14px !important;
    color: #111827 !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    background-color: #FFFFFF !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #9CA3AF !important; }

/* ── Selectbox ────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div:first-child {
    background-color: #F9FAFB !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #475569 !important;
    color: #475569 !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1) !important;
}
ul[role="listbox"] li,
[data-baseweb="menu"] li,
[data-baseweb="menu"] div,
ul[role="listbox"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
ul[role="listbox"] li:hover,
[data-baseweb="menu"] li:hover {
    background-color: #F1F5F9 !important;
}

/* ── Date Input ───────────────────────────────────────────────────── */
[data-testid="stDateInput"] input {
    background-color: #F9FAFB !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
[data-testid="stDateInput"] svg {
    fill: #475569 !important;
    color: #475569 !important;
}

/* ── Slider ───────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child > div {
    background: linear-gradient(90deg, #2563EB 0%, #2563EB 100%) !important;
}

/* ── Progress Bar ─────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
    border-radius: 999px !important;
}
.stProgress > div > div {
    border-radius: 999px !important;
    height: 6px !important;
    background-color: #E5E7EB !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #F3F4F6 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6B7280 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 0.4rem 0.875rem !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04) !important;
}
[data-testid="stTabContent"] { padding-top: 1rem !important; }

/* ── Expander ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
[data-testid="stExpander"] summary {
    padding: 0.875rem 1rem !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: #374151 !important;
}
[data-testid="stExpander"] summary:hover { color: #111827 !important; }
[data-testid="stExpander"] summary + div { padding: 1rem !important; }

/* ── Bordered Container (st.container(border=True)) ──────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 0 0 1px rgba(0,0,0,0.02) !important;
    padding: 0 !important;
    overflow: hidden !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.25rem 1.5rem !important;
}

/* ── Metric ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetric"] label,
[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] > div {
    color: #0A0A0A !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
}

/* ── Alerts ───────────────────────────────────────────────────────── */
[data-testid="stInfo"] {
    background-color: #EFF6FF !important;
    border: none !important;
    border-left: 4px solid #2563EB !important;
    border-radius: 8px !important;
    color: #1E40AF !important;
}
[data-testid="stSuccess"] {
    background-color: #F0FDF4 !important;
    border: none !important;
    border-left: 4px solid #16A34A !important;
    border-radius: 8px !important;
    color: #15803D !important;
}
[data-testid="stWarning"] {
    background-color: #FFFBEB !important;
    border: none !important;
    border-left: 4px solid #D97706 !important;
    border-radius: 8px !important;
    color: #92400E !important;
}
[data-testid="stError"] {
    background-color: #FEF2F2 !important;
    border: none !important;
    border-left: 4px solid #DC2626 !important;
    border-radius: 8px !important;
    color: #991B1B !important;
}

/* ── DataFrame ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E5E5 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}

/* ── Form ─────────────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* ── Camera Input ─────────────────────────────────────────────────── */
[data-testid="stCameraInput"] > div > div {
    border: 2px dashed #E5E5E5 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    transition: border-color 0.15s !important;
}
[data-testid="stCameraInput"] > div > div:hover {
    border-color: #2563EB !important;
}

/* ── Typography ───────────────────────────────────────────────────── */
h1 { color:#0A0A0A!important; font-weight:600!important; font-size:28px!important;
     letter-spacing:-0.025em!important; line-height:1.2!important; }
h2 { color:#111827!important; font-weight:600!important; font-size:20px!important;
     letter-spacing:-0.015em!important; }
h3 { color:#374151!important; font-weight:500!important; font-size:16px!important; }
p, [data-testid="stMarkdown"] p { color:#374151!important; font-size:14px!important; line-height:1.5!important; }
.stCaption > div, [data-testid="stCaptionContainer"] {
    color:#6B7280!important; font-size:12px!important; }
hr { border:none!important; border-top:1px solid #E5E5E5!important; margin:1.25rem 0!important; }
strong { color: inherit !important; }

/* ── Checkbox ─────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] > label { font-size:14px!important; color:#374151!important; }

/* ── Download Button ─────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: #FFFFFF !important;
    color: #374151 !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    height: 40px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #F9FAFB !important;
    border-color: #D1D5DB !important;
}
</style>
""", unsafe_allow_html=True)


def _page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page header for all pages."""
    sub = (
        f'<p style="font-size:14px;color:#6B7280;margin:6px 0 0 0;font-weight:400;">'
        f'{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:1.75rem;padding-bottom:1rem;'
        f'border-bottom:1px solid #F3F4F6;">'
        f'<h2 style="font-size:24px;font-weight:600;color:#0A0A0A;'
        f'letter-spacing:-0.025em;margin:0;line-height:1.2;">{title}</h2>'
        f'{sub}</div>',
        unsafe_allow_html=True,
    )


_inject_css()
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
if "page" not in st.session_state:
    st.session_state["page"] = ""
def _get_logo_b64() -> str | None:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "vignan_logo.jpg")
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None
    return None


# ===========================================================================
# LOGIN PAGE
# ===========================================================================
def show_login():
    # Center the card using 3-column layout
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        # ── Logo / branding ──────────────────────────────────────────
        logo_b64 = _get_logo_b64()
        logo_html = (
            f'<div style="margin-bottom:1rem;">'
            f'<img src="data:image/jpeg;base64,{logo_b64}" '
            f'style="max-width:320px; width:100%; height:auto; display:inline-block;" '
            f'alt="Vignan University" />'
            f'</div>'
            if logo_b64 else '<div style="font-size:48px; line-height:1; margin-bottom:0.75rem;">🎓</div>'
        )
        st.markdown(f"""
        <div style="text-align:center; padding:1.25rem 0 1.5rem 0;">
            {logo_html}
            <div style="font-size:24px; font-weight:700; color:#0A0A0A;
                        letter-spacing:-0.03em; line-height:1.2;">
                CampusAttend
            </div>
            <div style="font-size:13px; color:#6B7280; margin-top:4px;">
                Smart Campus Attendance System
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Login form ───────────────────────────────────────────────
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. staff1")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button(
                "Sign In", use_container_width=True, type="primary"
            )

        if submitted:
            role = auth.verify_login(username, password)
            if role:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = role
                # Set default landing page per role
                st.session_state["page"] = (
                    "Register Student" if role == "staff" else "Review & Post"
                )
                st.rerun()
            else:
                st.error("Invalid username or password.")

        # ── Demo credentials ─────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:1.5rem; padding-top:1.25rem; border-top:1px solid #F3F4F6;">
            <div style="font-size:11px; font-weight:600; color:#9CA3AF;
                        text-transform:uppercase; letter-spacing:0.06em;
                        margin-bottom:10px; text-align:center;">
                Demo Credentials
            </div>
            <div style="display:flex; flex-direction:column; gap:6px;">
                <div style="background:#F9FAFB; border:1px solid #E5E5E5; border-radius:8px;
                            padding:7px 12px; font-size:12px; color:#374151;
                            font-family:'Courier New',monospace; display:flex;
                            justify-content:space-between; align-items:center;">
                    <span style="color:#6B7280; font-size:11px; font-family:Inter,sans-serif;
                                 font-weight:500;">Staff</span>
                    <span>staff1 / staff123</span>
                </div>
                <div style="background:#F9FAFB; border:1px solid #E5E5E5; border-radius:8px;
                            padding:7px 12px; font-size:12px; color:#374151;
                            font-family:'Courier New',monospace; display:flex;
                            justify-content:space-between; align-items:center;">
                    <span style="color:#6B7280; font-size:11px; font-family:Inter,sans-serif;
                                 font-weight:500;">CR</span>
                    <span>cr1 / cr123</span>
                </div>
                <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
                            padding:7px 12px; font-size:12px; color:#1E40AF;
                            font-family:'Courier New',monospace; display:flex;
                            justify-content:space-between; align-items:center;">
                    <span style="color:#2563EB; font-size:11px; font-family:Inter,sans-serif;
                                 font-weight:500;">Faculty</span>
                    <span>faculty1 / fac123</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)



# ===========================================================================
# PAGE: Register Student  (staff only)
# ===========================================================================
def page_register():
    _page_header(
        "Student Registration",
        "Register a new student with 3 face photos for multi-angle biometric enrollment.",
    )
    col_info, col_cam = st.columns([1, 1], gap="large")
    with col_info:
        with st.container(border=True):
            st.markdown("**📋 Student Details**")
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
        with st.container(border=True):
            st.markdown("**📸 3-Photo Face Capture**")
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
    _page_header(
        "Live Face Attendance",
        "Select the class and snap a live photo — attendance is saved as Pending for faculty review.",
    )
    username = st.session_state["username"]
    with st.container(border=True):
        st.markdown("**🏫 Class Selection**")
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
        with st.container(border=True):
            st.markdown("**📸 Live Student Face**")
            att_photo = st.camera_input("Capture student face", key="att_camera")
            mark_clicked = st.button("📝 Mark Attendance", type="primary", use_container_width=True)
    with col_result:
        with st.container(border=True):
            st.markdown("**📊 Recognition Result**")
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
    _page_header(
        "Review & Post Attendance",
        "Review pending records, add or remove entries, then post to finalise.",
    )
    # ---- Filters ----
    with st.container(border=True):
        st.markdown("**🔍 Select Class & Date**")
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
    _page_header(
        "Attendance History",
        "Showing posted records." if role != "faculty" else "Full attendance log — posted and pending records.",
    )
    is_faculty = role == "faculty"
    # ---- Filters ----
    with st.container(border=True):
        st.markdown("**🔍 Filters**")
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

    # ── Build nav items per role ────────────────────────────────────
    if role == "staff":
        nav_items = [
            ("📋", "Register Student"),
            ("📸", "Take Attendance"),
            ("📊", "Attendance History"),
        ]
    else:
        nav_items = [
            ("✅", "Review & Post"),
            ("📊", "Attendance History"),
        ]

    page_names = [item[1] for item in nav_items]
    # Ensure current page is valid for this role
    if st.session_state.get("page") not in page_names:
        st.session_state["page"] = page_names[0]

    # ── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        # Branding
        _sb_logo_b64 = _get_logo_b64()
        _sb_logo_html = (
            f'<div style="margin-bottom:0.75rem;">'
            f'<img src="data:image/jpeg;base64,{_sb_logo_b64}" '
            f'style="max-width:180px; width:100%; height:auto;" alt="Vignan Logo" />'
            f'</div>'
            if _sb_logo_b64 else ""
        )
        st.markdown(f"""
        <div style="padding-bottom:1.25rem; border-bottom:1px solid #F3F4F6;
                    margin-bottom:1.25rem;">
            {_sb_logo_html}
            <div style="font-size:17px; font-weight:600; color:#0A0A0A;
                        letter-spacing:-0.015em;">
                🎓 CampusAttend
            </div>
            <div style="font-size:11px; color:#9CA3AF; margin-top:3px;
                        font-weight:400;">
                Smart Attendance System
            </div>
        </div>
        """, unsafe_allow_html=True)

        # User info pill
        avatar = username[0].upper() if username else "U"
        role_label = "Faculty" if role == "faculty" else "Staff"
        badge_bg = "#EFF6FF" if role == "faculty" else "#F0FDF4"
        badge_color = "#1D4ED8" if role == "faculty" else "#15803D"
        st.markdown(f"""
        <div style="background:#F9FAFB; border:1px solid #E5E5E5; border-radius:10px;
                    padding:10px 12px; margin-bottom:1.5rem;
                    display:flex; align-items:center; gap:10px;">
            <div style="width:32px; height:32px; background:#2563EB; border-radius:50%;
                        display:flex; align-items:center; justify-content:center;
                        color:white; font-size:13px; font-weight:600; flex-shrink:0;">
                {avatar}
            </div>
            <div>
                <div style="font-size:13px; font-weight:500; color:#111827;">
                    {username}
                </div>
                <div style="display:inline-block; margin-top:3px; padding:1px 8px;
                            background:{badge_bg}; color:{badge_color};
                            border-radius:999px; font-size:10px; font-weight:600;">
                    {role_label}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation label
        st.markdown("""
        <div style="font-size:10px; font-weight:600; color:#9CA3AF;
                    text-transform:uppercase; letter-spacing:0.08em;
                    padding:0 0.25rem; margin-bottom:6px;">
            Menu
        </div>
        """, unsafe_allow_html=True)

        # Nav items — active as styled div, inactive as button
        for icon, label in nav_items:
            if st.session_state["page"] == label:
                st.markdown(f"""
                <div style="background:#EFF6FF; border-radius:8px;
                            padding:0.5rem 0.75rem; margin-bottom:3px;
                            display:flex; align-items:center; gap:8px;
                            font-size:14px; font-weight:500; color:#2563EB;">
                    <span>{icon}</span><span>{label}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(
                    f"{icon}  {label}",
                    key=f"nav_{label}",
                    use_container_width=True,
                ):
                    st.session_state["page"] = label
                    st.rerun()

        # Logout
        st.markdown("""
        <div style="margin-top:1.5rem; padding-top:1rem;
                    border-top:1px solid #F3F4F6;">
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪  Sign Out", use_container_width=True, key="logout_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Page dispatch ────────────────────────────────────────────────
    page = st.session_state["page"]
    if page == "Register Student":
        page_register()
    elif page == "Take Attendance":
        page_take_attendance()
    elif page == "Review & Post":
        page_review_post()
    elif page == "Attendance History":
        page_history(role)
# To run type .\venv\Scripts\streamlit.exe run app.py
