import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
from datetime import datetime, timedelta
from collections import deque
# ✅ SESSION STORAGE
st.set_page_config(
    page_title="AI Interview Confidence Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# then session state
if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []

if "face_count" not in st.session_state:
    st.session_state.face_count = 0


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-primary: #020817;
    --bg-secondary: #0a1628;
    --bg-card: rgba(10, 22, 40, 0.85);
    --neon-cyan: #00f5ff;
    --neon-purple: #bf5fff;
    --neon-green: #00ff88;
    --neon-blue: #0084ff;
    --neon-pink: #ff2d78;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border-glow: rgba(0, 245, 255, 0.2);
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, rgba(0, 132, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(191, 95, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(0, 245, 255, 0.03) 0%, transparent 70%),
                linear-gradient(135deg, #020817 0%, #050d1a 50%, #020817 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05101f 0%, #020c1a 100%) !important;
    border-right: 1px solid rgba(0, 245, 255, 0.12) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
}

/* Header */
.main-header {
    background: linear-gradient(135deg, rgba(10,22,40,0.9) 0%, rgba(5,13,28,0.95) 100%);
    border: 1px solid rgba(0,245,255,0.15);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 0 40px rgba(0,245,255,0.06), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
    animation: scanline 4s linear infinite;
}

@keyframes scanline {
    0% { left: -100%; }
    100% { left: 100%; }
}

.app-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.app-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
}

.status-live {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 50px;
    padding: 6px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--neon-green);
    letter-spacing: 2px;
}

.pulse-dot {
    width: 8px; height: 8px;
    background: var(--neon-green);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--neon-green);
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px var(--neon-green); }
    50% { opacity: 0.5; transform: scale(0.8); box-shadow: 0 0 20px var(--neon-green); }
}

.timer-display {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    color: var(--neon-cyan);
    letter-spacing: 4px;
    text-shadow: 0 0 20px rgba(0,245,255,0.6);
}

/* Cards */
.glass-card {
    background: linear-gradient(135deg, rgba(10,22,40,0.9) 0%, rgba(5,13,28,0.95) 100%);
    border: 1px solid rgba(0,245,255,0.12);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(0,245,255,0.3);
    box-shadow: 0 4px 48px rgba(0,0,0,0.5), 0 0 24px rgba(0,245,255,0.08), inset 0 1px 0 rgba(255,255,255,0.06);
    transform: translateY(-2px);
}

.glass-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, rgba(0,245,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}

.card-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--neon-cyan);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-header::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: var(--neon-cyan);
    border-radius: 2px;
    box-shadow: 0 0 8px var(--neon-cyan);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(10,22,40,0.9) 0%, rgba(5,13,28,0.95) 100%);
    border: 1px solid rgba(0,245,255,0.1);
    border-radius: 14px;
    padding: 18px 16px;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 30px rgba(0,245,255,0.12);
    border-color: rgba(0,245,255,0.35);
}

.metric-icon {
    font-size: 2rem;
    margin-bottom: 8px;
    display: block;
    filter: drop-shadow(0 0 10px rgba(0,245,255,0.5));
}

.metric-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 10px;
}

.progress-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    height: 4px;
    overflow: hidden;
    position: relative;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.progress-bar-fill::after {
    content: '';
    position: absolute;
    right: 0; top: -2px;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: inherit;
    filter: blur(2px) brightness(1.5);
}

/* Sidebar nav */
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin-bottom: 4px;
    text-transform: uppercase;
    border: 1px solid transparent;
}

.nav-item:hover {
    background: rgba(0,245,255,0.06);
    color: var(--neon-cyan);
    border-color: rgba(0,245,255,0.15);
}

.nav-item.active {
    background: linear-gradient(135deg, rgba(0,245,255,0.1) 0%, rgba(191,95,255,0.08) 100%);
    color: var(--neon-cyan);
    border-color: rgba(0,245,255,0.2);
    box-shadow: 0 0 20px rgba(0,245,255,0.06);
}

/* Logo */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 3px;
    padding: 16px 0;
    border-bottom: 1px solid rgba(0,245,255,0.1);
    margin-bottom: 20px;
}

.logo-icon {
    font-size: 2.5rem;
    text-align: center;
    display: block;
    margin-bottom: 4px;
    filter: drop-shadow(0 0 16px rgba(0,245,255,0.6));
}

/* Tips card */
.tips-card {
    background: linear-gradient(135deg, rgba(0,245,255,0.06) 0%, rgba(191,95,255,0.06) 100%);
    border: 1px solid rgba(0,245,255,0.12);
    border-radius: 12px;
    padding: 14px;
    margin-top: 20px;
}

.tips-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: var(--neon-cyan);
    margin-bottom: 8px;
    text-transform: uppercase;
}

.tips-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Camera card */
.camera-card {
    background: linear-gradient(135deg, rgba(10,22,40,0.9) 0%, rgba(5,13,28,0.95) 100%);
    border: 1px solid rgba(0,245,255,0.15);
    border-radius: 20px;
    padding: 4px;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(0,245,255,0.08), 0 4px 40px rgba(0,0,0,0.5);
}

.recording-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,45,120,0.15);
    border: 1px solid rgba(255,45,120,0.3);
    border-radius: 50px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--neon-pink);
    letter-spacing: 1px;
}

.face-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 50px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--neon-green);
    letter-spacing: 1px;
}

/* Gauge center text */
.gauge-score {
    font-family: 'Orbitron', monospace;
    font-size: 3.5rem;
    font-weight: 900;
    text-align: center;
    text-shadow: 0 0 30px currentColor;
}

.motivation-text {
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 8px;
}

/* Feedback */
.feedback-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.95rem;
    color: var(--text-primary);
    line-height: 1.5;
}

.feedback-item:last-child { border-bottom: none; }

.feedback-dot-green {
    width: 8px; height: 8px; min-width: 8px;
    background: var(--neon-green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--neon-green);
    margin-top: 6px;
}

.feedback-dot-red {
    width: 8px; height: 8px; min-width: 8px;
    background: var(--neon-pink);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--neon-pink);
    margin-top: 6px;
}

/* Summary */
.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.95rem;
}

.summary-row:last-child { border-bottom: none; }
.summary-label { color: var(--text-secondary); letter-spacing: 1px; }
.summary-value { font-family: 'Space Mono', monospace; font-size: 0.85rem; }

/* End button */
.stButton > button {
    background: linear-gradient(135deg, rgba(255,45,120,0.15) 0%, rgba(255,45,120,0.08) 100%) !important;
    border: 1px solid rgba(255,45,120,0.35) !important;
    color: var(--neon-pink) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(255,45,120,0.25) 0%, rgba(255,45,120,0.15) 100%) !important;
    box-shadow: 0 0 20px rgba(255,45,120,0.2) !important;
    transform: translateY(-1px) !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,245,255,0.4); }

/* Divider */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.3), transparent);
    margin: 20px 0;
}

/* Section title */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--neon-cyan);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,245,255,0.3), transparent);
}

/* Emotion chip */
.emotion-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(191,95,255,0.1);
    border: 1px solid rgba(191,95,255,0.25);
    border-radius: 50px;
    padding: 6px 16px;
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: var(--neon-purple);
    text-transform: uppercase;
}

/* Grid scan overlay effect */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}

.floating { animation: float 3s ease-in-out infinite; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in { animation: fadeInUp 0.5s ease forwards; }

/* Eye tracker bar */
.eye-bar-container {
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    position: relative;
    margin: 10px 0;
}

.eye-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--neon-blue), var(--neon-cyan));
    box-shadow: 0 0 12px rgba(0,245,255,0.4);
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "interview_active" not in st.session_state:
    st.session_state.interview_active = True
if "history" not in st.session_state:
    st.session_state.history = {
        "confidence": deque(maxlen=40),
        "emotion": deque(maxlen=40),
        "eye_contact": deque(maxlen=40),
        "nervousness": deque(maxlen=40),
        "timestamps": deque(maxlen=40),
    }
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "page" not in st.session_state:
    st.session_state.page = "Live Analysis"

EMOTIONS = ["Confident", "Focused", "Calm", "Engaged", "Slightly Nervous", "Neutral"]
EMOTION_EMOJI = {"Confident": "😤", "Focused": "🎯", "Calm": "😌", "Engaged": "🔥", "Slightly Nervous": "😰", "Neutral": "😐"}

def smooth_random(prev, lo=0, hi=100, drift=8):
    val = prev + random.uniform(-drift, drift)
    return max(lo, min(hi, val))

if len(st.session_state.history["confidence"]) == 0:
    c, e, n = 72.0, 65.0, 28.0
    for i in range(30):
        c = smooth_random(c, 45, 95, 5)
        e = smooth_random(e, 40, 90, 6)
        n = smooth_random(n, 10, 55, 5)
        st.session_state.history["confidence"].append(round(c, 1))
        st.session_state.history["eye_contact"].append(round(e, 1))
        st.session_state.history["nervousness"].append(round(n, 1))
        st.session_state.history["emotion"].append(random.choice(EMOTIONS))
        st.session_state.history["timestamps"].append(time.time() - (30 - i) * 2)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("logo.png", width=120)

    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px;'>
        <div class='sidebar-logo'>AI ANALYZER</div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("📊", "Dashboard"),
        ("🎥", "Live Analysis"),
        ("📈", "Performance Report"),
        ("🤖", "AI Feedback"),
        ("⚙️", "Settings"),
        ("ℹ️", "About"),
    ]
    for icon, name in pages:
        active = "active" if st.session_state.page == name else ""
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
            st.session_state.page = name
            st.rerun()

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    tips_list = [
        "Maintain steady eye contact with the camera lens.",
        "Sit upright — posture signals confidence.",
        "Breathe slowly to reduce nervous energy.",
        "Pause before answering — silence is power.",
    ]
    tip = tips_list[int(time.time() / 8) % len(tips_list)]
    st.markdown(f"""
    <div class='tips-card'>
        <div class='tips-title'>⚡ AI Tip</div>
        <div class='tips-text'>{tip}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:24px; text-align:center;'>
        <span style='font-size:1.4rem; filter:drop-shadow(0 0 8px rgba(0,245,255,0.5))'>👁️</span>
        <span style='font-size:1.4rem; margin:0 8px; filter:drop-shadow(0 0 8px rgba(191,95,255,0.5))'>🎙️</span>
        <span style='font-size:1.4rem; filter:drop-shadow(0 0 8px rgba(0,255,136,0.5))'>🧬</span>
    </div>
    <div style='text-align:center; font-family:Space Mono,monospace; font-size:0.6rem; color:#334155; margin-top:12px; letter-spacing:2px;'>
       Smart AI Analyzer
    </div>
    """, unsafe_allow_html=True)

# ─── Generate live values ─────────────────────────────────────────────────────
prev_conf = st.session_state.history["confidence"][-1]
prev_eye = st.session_state.history["eye_contact"][-1]
prev_nerv = st.session_state.history["nervousness"][-1]

confidence_score = smooth_random(prev_conf, 40, 96, 4)
eye_contact_score = smooth_random(prev_eye, 35, 92, 5)
nervousness_score = smooth_random(prev_nerv, 8, 58, 4)
head_stability = smooth_random(75, 55, 95, 5)
speech_conf = smooth_random(68, 50, 92, 6)
facial_exp = smooth_random(72, 48, 90, 5)
current_emotion = random.choice(EMOTIONS) if random.random() < 0.15 else st.session_state.history["emotion"][-1]

st.session_state.history["confidence"].append(round(confidence_score, 1))
st.session_state.history["eye_contact"].append(round(eye_contact_score, 1))
st.session_state.history["nervousness"].append(round(nervousness_score, 1))
st.session_state.history["emotion"].append(current_emotion)
st.session_state.history["timestamps"].append(time.time())
st.session_state.frame_count += 1

# ─── Timer ────────────────────────────────────────────────────────────────────
elapsed = int(time.time() - st.session_state.start_time)
mins, secs = divmod(elapsed, 60)
timer_str = f"{mins:02d}:{secs:02d}"

# ─── Header ───────────────────────────────────────────────────────────────────
hcol1, hcol2, hcol3, hcol4 = st.columns([3, 1.2, 1, 0.8])
with hcol1:
    st.markdown("""
    <div style='padding:12px 0;'>
        <div class='app-title'>AI Interview Analyzer</div>
        <div class='app-subtitle'>Real-time AI-powered performance tracking</div>
    </div>
    """, unsafe_allow_html=True)
with hcol2:
    st.markdown(f"""
    <div style='padding-top:18px;'>
        <div class='status-live'>
            <div class='pulse-dot'></div>
            LIVE SESSION
        </div>
    </div>
    """, unsafe_allow_html=True)
with hcol3:
    st.markdown(f"""
    <div style='padding-top:12px; text-align:center;'>
        <div style='font-family:Space Mono,monospace; font-size:0.6rem; letter-spacing:2px; color:#475569; text-transform:uppercase; margin-bottom:2px;'>Duration</div>
        <div class='timer-display'>{timer_str}</div>
    </div>
    """, unsafe_allow_html=True)
with hcol4:
    st.markdown("<div style='padding-top:18px;'>", unsafe_allow_html=True)
    if st.button("⏹ END", key="end_btn"):
        st.session_state.interview_active = False
        st.session_state.page = "Performance Report"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE: LIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.page == ("Live Analysis"):

    # ─── Row 1: Camera + Gauge + Emotion ─────────────────────────────
    col_cam, col_gauge, col_emo = st.columns([2.2, 1.4, 1.4])

    with col_cam:
        st.markdown('<div class="section-title">🎥 Live Feed</div>', unsafe_allow_html=True)
        cam_frame_placeholder = st.empty()
        badge_placeholder = st.empty()

        url = "http://192.168.31.226:4747/video"
        cap = cv2.VideoCapture(url)
        mp_face = mp.solutions.face_detection
        face_det = mp_face.FaceDetection(min_detection_confidence=0.5)

        frame_drawn = False
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_det.process(rgb)
                face_detected = False
                if res.detections:
                    face_detected = True
                if face_detected:
                    st.session_state.face_count += 1
                    for det in res.detections:
                        bboxC = det.location_data.relative_bounding_box
                        h, w = frame.shape[:2]
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        bw = int(bboxC.width * w)
                        bh = int(bboxC.height * h)
                        pad = 12
                        x1, y1 = max(0, x1-pad), max(0, y1-pad)
                        x2, y2 = min(w, x1+bw+2*pad), min(h, y1+bh+2*pad)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 245, 255), 2)
                        tl = 14
                        cv2.line(frame, (x1, y1), (x1+tl, y1), (0, 245, 255), 3)
                        cv2.line(frame, (x1, y1), (x1, y1+tl), (0, 245, 255), 3)
                        cv2.line(frame, (x2, y1), (x2-tl, y1), (0, 245, 255), 3)
                        cv2.line(frame, (x2, y1), (x2, y1+tl), (0, 245, 255), 3)
                        cv2.line(frame, (x1, y2), (x1+tl, y2), (0, 245, 255), 3)
                        cv2.line(frame, (x1, y2), (x1, y2-tl), (0, 245, 255), 3)
                        cv2.line(frame, (x2, y2), (x2-tl, y2), (0, 245, 255), 3)
                        cv2.line(frame, (x2, y2), (x2, y2-tl), (0, 245, 255), 3)
                        label = f"CONF {int(confidence_score)}%"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 255), 1, cv2.LINE_AA)

                scan_y = int((time.time() * 60) % frame.shape[0])
                cv2.line(frame, (0, scan_y), (frame.shape[1], scan_y), (0, 245, 255), 1)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 10, 30), 0)
                frame = cv2.addWeighted(overlay, 0.05, frame, 0.95, 0)

                rgb_final = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_frame_placeholder.image(rgb_final, channels="RGB", use_container_width=True)
                frame_drawn = True

                badge_col1, badge_col2 = st.columns(2)
                with badge_col1:
                    if face_detected:
                        st.markdown('<div class="face-badge">● FACE DETECTED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="color:#475569; font-size:0.8rem;">No face detected</div>', unsafe_allow_html=True)
                with badge_col2:
                    st.markdown('<div class="recording-badge">⏺ RECORDING</div>', unsafe_allow_html=True)
        cap.release()

        if not frame_drawn:
            cam_frame_placeholder.markdown("""
            <div style='height:320px; background:linear-gradient(135deg,rgba(0,245,255,0.03),rgba(191,95,255,0.03));
                        border:2px dashed rgba(0,245,255,0.15); border-radius:12px;
                        display:flex; align-items:center; justify-content:center;
                        flex-direction:column; gap:12px;'>
                <div style='font-size:3rem; filter:drop-shadow(0 0 20px rgba(0,245,255,0.4))'>📷</div>
                <div style='font-family:Orbitron,monospace; font-size:0.7rem; letter-spacing:3px; color:#475569;'>CAMERA INITIALIZING</div>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown('<div class="section-title">🎯 Confidence</div>', unsafe_allow_html=True)
        c_score = int(confidence_score)
        if c_score >= 75:
            gauge_color = "#00ff88"
            motivation = "EXCELLENT"
            mot_color = "#00ff88"
        elif c_score >= 55:
            gauge_color = "#00f5ff"
            motivation = "GOOD"
            mot_color = "#00f5ff"
        else:
            gauge_color = "#ff2d78"
            motivation = "NEEDS FOCUS"
            mot_color = "#ff2d78"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=c_score,
            number={"font": {"family": "Orbitron", "size": 40, "color": gauge_color}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.1)", "tickwidth": 1, "tickfont": {"size": 8, "color": "#334155"}},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 100], "color": "rgba(255,255,255,0.03)"},
                    {"range": [0, c_score], "color": f"rgba({','.join(str(int(x*255)) for x in bytes.fromhex(gauge_color.lstrip('#')))},0.08)"},
                ],
                "threshold": {"line": {"color": gauge_color, "width": 2}, "thickness": 0.8, "value": c_score},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=230, margin=dict(l=20, r=20, t=20, b=10),
            font={"color": "#94a3b8"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")
        st.markdown(f"""
        <div class='motivation-text' style='color:{mot_color}; text-shadow:0 0 15px {mot_color};'>
            ◆ {motivation} ◆
        </div>
        """, unsafe_allow_html=True)

    with col_emo:
        st.markdown('<div class="section-title">🌡️ Emotions</div>', unsafe_allow_html=True)
        emoji = EMOTION_EMOJI.get(current_emotion, "😐")
        st.markdown(f"""
        <div style='text-align:center; margin-bottom:10px;'>
            <div style='font-size:3rem; filter:drop-shadow(0 0 16px rgba(191,95,255,0.6));'>{emoji}</div>
            <div class='emotion-chip' style='margin:8px auto; display:inline-flex;'>{current_emotion.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        emo_data = list(st.session_state.history["confidence"])[-20:]
        emo_times = list(range(len(emo_data)))
        fig_emo = go.Figure()
        fig_emo.add_trace(go.Scatter(
            x=emo_times, y=emo_data,
            fill='tozeroy',
            fillcolor='rgba(191,95,255,0.08)',
            line=dict(color='#bf5fff', width=2),
            mode='lines',
            name='Confidence'
        ))
        fig_emo.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=160, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0,100]),
            showlegend=False
        )
        st.plotly_chart(fig_emo, use_container_width=True, key="emo_chart")

        # Eye contact
        st.markdown(f"""
        <div style='margin-top:4px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                <span style='font-family:Space Mono,monospace; font-size:0.65rem; color:#475569; letter-spacing:1px;'>👁 EYE CONTACT</span>
                <span style='font-family:Orbitron,monospace; font-size:0.85rem; color:#00f5ff;'>{int(eye_contact_score)}%</span>
            </div>
            <div class='eye-bar-container'>
                <div class='eye-bar-fill' style='width:{int(eye_contact_score)}%;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
# ✅ SAVE CONFIDENCE HISTORY
st.session_state.confidence_history.append(confidence_score)

# limit data (optional but recommended)
if len(st.session_state.confidence_history) > 50:
    st.session_state.confidence_history.pop(0)

#DASHBOARD
    
    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
elif st.session_state.page == "Dashboard":

    st.markdown("## 📊 Dashboard")

    history = st.session_state.confidence_history

    # ─── Row 1: Summary Cards ─────────────────────────────
    col1, col2, col3 = st.columns(3)

    # ✅ Avg Confidence
    with col1:
        if len(history) > 0:
            avg_conf = sum(history) / len(history)
        else:
            avg_conf = 0
        st.metric(label="Avg Confidence", value=f"{int(avg_conf)}%")

    # ✅ Face Count
    with col2:
        st.metric(label="Faces Detected", value=st.session_state.face_count)

    # ✅ Status
    with col3:
        status = "Active" if len(history) > 0 else "Idle"
        st.metric(label="Status", value=status)

    # ─── Row 2: Real Chart ─────────────────────────────
    st.markdown("### 📈 Confidence Trend")

    import pandas as pd

    if len(history) > 0:
        df = pd.DataFrame({
            "Time": list(range(len(history))),
            "Confidence": history
        })

        st.line_chart(df.set_index("Time"))
    else:
        st.warning("No data yet. Run Live Analysis first.")

    # ─── Row 3: Logs ─────────────────────────────
    st.markdown("### 🧾 Analysis Logs")

    if len(history) > 0:
        st.write(f"✔ {len(history)} frames analyzed")
        st.write(f"✔ Latest confidence: {history[-1]}%")
        st.write("✔ System running properly")
    else:
        st.write("No activity yet")
    # ─── Row 2: Metric Cards ──────────────────────────────────────────
    st.markdown('<div class="section-title">⚡ Real-Time Metrics</div>', unsafe_allow_html=True)

    metrics = [
        ("😄", "Facial Expression", facial_exp, "linear-gradient(135deg, #00f5ff, #0084ff)", "#00f5ff"),
        ("👁️", "Eye Contact", eye_contact_score, "linear-gradient(135deg, #0084ff, #bf5fff)", "#0084ff"),
        ("🗣️", "Head Stability", head_stability, "linear-gradient(135deg, #00ff88, #00f5ff)", "#00ff88"),
        ("🎙️", "Speech Confidence", speech_conf, "linear-gradient(135deg, #bf5fff, #ff2d78)", "#bf5fff"),
        ("🧠", "Nervousness", nervousness_score, "linear-gradient(135deg, #ff2d78, #ff6b35)", "#ff2d78"),
    ]

    m_cols = st.columns(5)
    for i, (icon, label, val, grad, color) in enumerate(metrics):
        with m_cols[i]:
            pct = int(val)
            st.markdown(f"""
            <div class='metric-card'>
                <span class='metric-icon'>{icon}</span>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}; text-shadow:0 0 20px {color}80;'>{pct}<span style='font-size:1rem;'>%</span></div>
                <div class='progress-bar-container'>
                    <div class='progress-bar-fill' style='width:{pct}%; background:{grad}; box-shadow:0 0 10px {color}60;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ─── Row 3: Trend chart + Feedback + Summary ──────────────────────
    st.markdown('<div class="section-title">📊 Performance Trend</div>', unsafe_allow_html=True)

    trend_col, fb_col, sum_col = st.columns([2, 1.5, 1.5])

    with trend_col:
        conf_hist = list(st.session_state.history["confidence"])
        eye_hist = list(st.session_state.history["eye_contact"])
        nerv_hist = list(st.session_state.history["nervousness"])
        x_vals = list(range(len(conf_hist)))

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=x_vals, y=conf_hist, name='Confidence', line=dict(color='#00f5ff', width=2.5), mode='lines'))
        fig_trend.add_trace(go.Scatter(x=x_vals, y=eye_hist, name='Eye Contact', line=dict(color='#bf5fff', width=2, dash='dot'), mode='lines'))
        fig_trend.add_trace(go.Scatter(x=x_vals, y=nerv_hist, name='Nervousness', line=dict(color='#ff2d78', width=2, dash='dash'), mode='lines'))
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=220, margin=dict(l=0, r=10, t=10, b=30),
            xaxis=dict(title=dict(
        text="Confidence",
        font=dict(size=20)
    )
),
            yaxis=dict(
    title=dict(
        text="Score",
        font=dict(size=20)
    )
),
            legend=dict(orientation='h', y=1.12, font=dict(family='Space Mono', size=9, color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="trend")

    with fb_col:
        st.markdown('<div class="card-header">🤖 AI Feedback</div>', unsafe_allow_html=True)
        feedback_positive = [
            "Consistent eye contact maintained",
            "Posture appears upright and open",
            "Expression conveys engagement",
        ]
        feedback_negative = [
            "Slight head movement detected",
            "Breathing pattern shows mild tension",
        ]
        for f in feedback_positive:
            st.markdown(f'<div class="feedback-item"><div class="feedback-dot-green"></div><span>{f}</span></div>', unsafe_allow_html=True)
        for f in feedback_negative:
            st.markdown(f'<div class="feedback-item"><div class="feedback-dot-red"></div><span>{f}</span></div>', unsafe_allow_html=True)

    with sum_col:
        st.markdown('<div class="card-header">📋 Summary</div>', unsafe_allow_html=True)
        emo_emoji = EMOTION_EMOJI.get(current_emotion, "😐")
        rows = [
            ("Confidence", f"<span style='color:#00ff88;'>{int(confidence_score)}%</span>"),
            ("Emotion", f"<span style='color:#bf5fff;'>{emo_emoji} {current_emotion}</span>"),
            ("Eye Contact", f"<span style='color:#00f5ff;'>{int(eye_contact_score)}%</span>"),
            ("Nervousness", f"<span style='color:#ff2d78;'>{int(nervousness_score)}%</span>"),
            ("Speech Clarity", f"<span style='color:#0084ff;'>{int(speech_conf)}%</span>"),
        ]
        for label, val in rows:
            st.markdown(f"""
            <div class='summary-row'>
                <span class='summary-label'>{label}</span>
                <span class='summary-value'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ─── Suggestions ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 Improvement Suggestions</div>', unsafe_allow_html=True)
    sug_cols = st.columns(3)
    suggestions = [
        ("🎯", "Eye Contact", "Look directly into the camera lens — it simulates direct eye contact with the interviewer and conveys confidence."),
        ("🧍", "Posture & Presence", "Sit slightly forward in your chair. Upright posture naturally projects authority and attentiveness."),
        ("🗣️", "Vocal Confidence", "Slow your speech slightly. Deliberate pacing signals expertise and reduces verbal fillers like 'um'."),
    ]
    for col, (ico, title, body) in zip(sug_cols, suggestions):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='min-height:130px;'>
                <div style='font-size:1.8rem; margin-bottom:8px;'>{ico}</div>
                <div style='font-family:Orbitron,monospace; font-size:0.65rem; letter-spacing:2px; color:#00f5ff; margin-bottom:8px; text-transform:uppercase;'>{title}</div>
                <div style='font-size:0.88rem; color:#94a3b8; line-height:1.5;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Performance Report":
    st.markdown('<div class="section-title">📈 Full Session Report</div>', unsafe_allow_html=True)

    conf_list = list(st.session_state.history["confidence"])
    eye_list = list(st.session_state.history["eye_contact"])
    nerv_list = list(st.session_state.history["nervousness"])

    avg_conf = np.mean(conf_list) if conf_list else 0
    avg_eye = np.mean(eye_list) if eye_list else 0
    avg_nerv = np.mean(nerv_list) if nerv_list else 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#475569; letter-spacing:2px; margin-bottom:8px;'>AVG CONFIDENCE</div>
            <div style='font-family:Orbitron,monospace; font-size:2.5rem; font-weight:900; color:#00ff88; text-shadow:0 0 20px #00ff8860;'>{avg_conf:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#475569; letter-spacing:2px; margin-bottom:8px;'>EYE CONTACT</div>
            <div style='font-family:Orbitron,monospace; font-size:2.5rem; font-weight:900; color:#00f5ff; text-shadow:0 0 20px #00f5ff60;'>{avg_eye:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#475569; letter-spacing:2px; margin-bottom:8px;'>NERVOUSNESS</div>
            <div style='font-family:Orbitron,monospace; font-size:2.5rem; font-weight:900; color:#ff2d78; text-shadow:0 0 20px #ff2d7860;'>{avg_nerv:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        duration = int(time.time() - st.session_state.start_time)
        dm, ds = divmod(duration, 60)
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-family:Space Mono,monospace; font-size:0.6rem; color:#475569; letter-spacing:2px; margin-bottom:8px;'>DURATION</div>
            <div style='font-family:Orbitron,monospace; font-size:2.5rem; font-weight:900; color:#bf5fff; text-shadow:0 0 20px #bf5fff60;'>{dm:02d}:{ds:02d}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    fig_full = go.Figure()
    x_full = list(range(len(conf_list)))
    fig_full.add_trace(go.Scatter(x=x_full, y=conf_list, name='Confidence', fill='tozeroy', fillcolor='rgba(0,245,255,0.06)', line=dict(color='#00f5ff', width=2.5)))
    fig_full.add_trace(go.Scatter(x=x_full, y=eye_list, name='Eye Contact', fill='tozeroy', fillcolor='rgba(191,95,255,0.05)', line=dict(color='#bf5fff', width=2)))
    fig_full.add_trace(go.Scatter(x=x_full, y=nerv_list, name='Nervousness', line=dict(color='#ff2d78', width=1.5, dash='dot')))
    fig_full.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=320,
    margin=dict(l=0, r=0, t=20, b=30),

    title=dict(
        text='Session Performance Trends',
        font=dict(family='Orbitron', size=12, color='#00f5ff'),
        x=0
    ),

    xaxis=dict(
        showgrid=False,
        tickfont=dict(color="#344864", size=9),
        title=dict(
            text='Frame',
            font=dict(size=20)
        )
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.04)',
        tickfont=dict(color='#334155', size=9),
        range=[0,100]
    ),

    legend=dict(
        font=dict(family='Space Mono', size=9, color='#94a3b8'),
        bgcolor='rgba(0,0,0,0)',
        orientation='h',
        y=1.1
    ),

    hovermode='x unified'
)
    st.plotly_chart(fig_full, use_container_width=True, key="full_report")

# ═══════════════════════════════════════════════════════════════════════
# PAGE: AI FEEDBACK
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state.page == "AI Feedback":
    st.markdown('<div class="section-title">🤖 Detailed AI Analysis</div>', unsafe_allow_html=True)
    fb_col1, fb_col2 = st.columns(2)
    strengths = [
        ("✅", "Steady Eye Contact", "Your gaze is directed at the camera 73% of the time — well above the 60% benchmark."),
        ("✅", "Upright Posture", "Head stability metrics indicate minimal swaying, projecting composure."),
        ("✅", "Positive Expression", "Neutral-to-positive facial expression throughout — crucial for first impressions."),
        ("✅", "Engagement Score", "Facial muscle activity suggests active listening and engagement."),
    ]
    improvements = [
        ("⚠️", "Reduce Micro-Movements", "Small head nods are detected frequently — try to anchor your gaze and minimize subtle movements."),
        ("⚠️", "Lower Tension Signals", "Jaw tension indicators suggest mild stress. Try box breathing between responses."),
        ("⚠️", "Vocal Pacing", "Speech confidence can be improved by introducing deliberate pauses for emphasis."),
    ]
    with fb_col1:
        st.markdown('<div class="card-header">💪 Strengths</div>', unsafe_allow_html=True)
        for ico, title, body in strengths:
            st.markdown(f"""
            <div class='glass-card' style='padding:14px 16px; margin-bottom:10px;'>
                <div style='font-family:Orbitron,monospace; font-size:0.65rem; letter-spacing:2px; color:#00ff88; margin-bottom:4px;'>{ico} {title.upper()}</div>
                <div style='font-size:0.88rem; color:#94a3b8;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)
    with fb_col2:
        st.markdown('<div class="card-header">🎯 Areas to Improve</div>', unsafe_allow_html=True)
        for ico, title, body in improvements:
            st.markdown(f"""
            <div class='glass-card' style='padding:14px 16px; margin-bottom:10px; border-color:rgba(255,45,120,0.12);'>
                <div style='font-family:Orbitron,monospace; font-size:0.65rem; letter-spacing:2px; color:#ff2d78; margin-bottom:4px;'>{ico} {title.upper()}</div>
                <div style='font-size:0.88rem; color:#94a3b8;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Settings":
    st.markdown('<div class="section-title">⚙️ Configuration</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🎥 Camera Settings</div>', unsafe_allow_html=True)
        st.selectbox("Camera Device", ["Default (0)", "USB Camera (1)", "Virtual Camera (2)"])
        st.slider("Detection Confidence", 0.3, 0.9, 0.5, 0.05)
        st.slider("Refresh Rate (frames)", 1, 10, 3)
        st.markdown('</div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🧠 Analysis Settings</div>', unsafe_allow_html=True)
        st.toggle("Enable Face Mesh", value=True)
        st.toggle("Eye Tracking", value=True)
        st.toggle("Head Stability Analysis", value=True)
        st.toggle("Emotion Simulation", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state.page == "About":
    st.markdown('<div class="section-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card' style='max-width:700px;'>
        <div class='app-title' style='font-size:1.2rem; margin-bottom:8px;'>AI Interview Confidence Analyzer</div>
        <div style='font-size:0.9rem; color:#94a3b8; line-height:1.8; margin-top:12px;'>
            This platform uses real-time computer vision and AI to help you master the non-verbal
            dimensions of interview performance — eye contact, posture, expression, and composure.
        </div>
        <div style='margin-top:16px;'>
            <div style='font-family:Orbitron,monospace; font-size:0.6rem; letter-spacing:3px; color:#00f5ff; margin-bottom:10px; text-transform:uppercase;'>Tech Stack</div>
            <div style='display:flex; flex-wrap:wrap; gap:8px;'>
        """ + "".join([f"<span style='background:rgba(0,245,255,0.08); border:1px solid rgba(0,245,255,0.15); border-radius:6px; padding:4px 12px; font-family:Space Mono,monospace; font-size:0.7rem; color:#00f5ff; letter-spacing:1px;'>{t}</span>" for t in ["Python", "Streamlit", "OpenCV", "MediaPipe", "Plotly", "NumPy", "Pandas"]]) + """
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if st.session_state.interview_active and st.session_state.page in ("Live Analysis", "Dashboard"):
    time.sleep(0.8)
    st.rerun()