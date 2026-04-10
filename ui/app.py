# FILE: customer_support_rag/ui/app.py
"""Streamlit demo UI for the Multi-Agent RAG Customer Support System.

Communicates with the FastAPI backend via HTTP (httpx).
Start the backend first: uvicorn api.main:app --reload --port 8000
Then run:             streamlit run ui/app.py
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
import os
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support RAG · Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")  # configurable via .env
TIMEOUT  = 120.0   # seconds — LLM calls can be slow

INTENT_META: dict[str, dict[str, str]] = {
    "refund":    {"emoji": "💸", "color": "#f97316"},
    "technical": {"emoji": "🔧", "color": "#3b82f6"},
    "billing":   {"emoji": "🧾", "color": "#8b5cf6"},
    "general":   {"emoji": "💬", "color": "#10b981"},
    "escalate":  {"emoji": "🚨", "color": "#ef4444"},
    "":          {"emoji": "❓", "color": "#6b7280"},
}

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ──────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── Cards ───────────────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}

.card-user {
    background: rgba(99,102,241,0.18);
    border-color: rgba(99,102,241,0.30);
    margin-left: 10%;
}

.card-bot {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.22);
    margin-right: 10%;
}

.card-hitl {
    background: rgba(245,158,11,0.14);
    border-color: rgba(245,158,11,0.35);
}

.card-error {
    background: rgba(239,68,68,0.14);
    border-color: rgba(239,68,68,0.30);
}

/* ── Badge ───────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* ── Meta row ────────────────────────────────────────────── */
.meta-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 10px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.55);
}

.meta-chip {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 2px 8px;
}

/* ── Scrollable chat area ─────────────────────────────────── */
.chat-scroll {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 4px;
}

/* ── Status dot ─────────────────────────────────────────── */
.dot-green  { color: #10b981; font-size: 1.1rem; }
.dot-red    { color: #ef4444; font-size: 1.1rem; }
.dot-yellow { color: #f59e0b; font-size: 1.1rem; }

/* ── Headings ─────────────────────────────────────────────── */
h1, h2, h3 { color: #f8fafc !important; }

/* ── Input box ───────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* ── Code block ──────────────────────────────────────────── */
code {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
    color: #a5f3fc !important;
}

/* ── Metric ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #f8fafc !important; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.55) !important; }

/* ── Info / warning / success boxes ─────────────────────── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "session_id"  not in st.session_state: st.session_state.session_id  = str(uuid.uuid4())
if "hitl_draft"  not in st.session_state: st.session_state.hitl_draft  = None
if "api_healthy" not in st.session_state: st.session_state.api_healthy = None
if "stats"       not in st.session_state: st.session_state.stats = {"total": 0, "hitl": 0, "errors": 0}
if "ingest_job"  not in st.session_state: st.session_state.ingest_job  = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs) -> tuple[int, dict]:
    """Thin HTTP wrapper — returns (status_code, body_dict)."""
    try:
        r = httpx.request(method, f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs)
        return r.status_code, r.json()
    except httpx.ConnectError:
        return 0, {"error": "Cannot connect to API. Is the backend running on :8000?"}
    except Exception as exc:
        return -1, {"error": str(exc)}


def check_health() -> bool:
    code, body = api("GET", "/health")
    return code == 200 and body.get("status") == "ok"


def intent_badge(intent: str) -> str:
    meta = INTENT_META.get(intent, INTENT_META[""])
    color = meta["color"]
    emoji = meta["emoji"]
    return (
        f'<span class="badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;">{emoji} {intent or "unknown"}</span>'
    )


def render_message(msg: dict) -> None:
    """Render one chat turn in the conversation view."""
    role = msg["role"]

    if role == "user":
        st.markdown(f"""
        <div class="card card-user">
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.45);margin-bottom:6px;">
                👤 You
            </div>
            <div style="color:#f8fafc;font-size:0.97rem;line-height:1.6;">
                {msg["content"]}
            </div>
        </div>""", unsafe_allow_html=True)

    elif role == "assistant":
        meta = msg.get("meta", {})
        intent = meta.get("intent", "")
        retries = meta.get("retry_count", 0)
        latency = meta.get("latency_s", None)

        latency_str = f'<span class="meta-chip">⏱ {latency:.1f}s</span>' if latency else ""
        retry_str   = (f'<span class="meta-chip" style="color:#f59e0b;">🔄 {retries} retr{"y" if retries==1 else "ies"}</span>'
                       if retries > 0 else "")

        st.markdown(f"""
        <div class="card card-bot">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-size:0.8rem;color:rgba(255,255,255,0.45);">🤖 Support Agent</span>
                {intent_badge(intent)}
            </div>
            <div style="color:#f0fdf4;font-size:0.97rem;line-height:1.7;">
                {msg["content"].replace(chr(10), "<br>")}
            </div>
            <div class="meta-row">
                {latency_str}
                {retry_str}
            </div>
        </div>""", unsafe_allow_html=True)

    elif role == "hitl":
        st.markdown(f"""
        <div class="card card-hitl">
            <div style="font-size:0.85rem;font-weight:600;color:#f59e0b;margin-bottom:8px;">
                ⏸ Human Review Required
            </div>
            <div style="color:#fef3c7;font-size:0.9rem;line-height:1.6;margin-bottom:6px;">
                <strong>Draft reply:</strong><br>{msg["content"].replace(chr(10), "<br>")}
            </div>
            <div style="font-size:0.78rem;color:rgba(255,255,255,0.45);">
                Submit feedback below to continue
            </div>
        </div>""", unsafe_allow_html=True)

    elif role == "error":
        st.markdown(f"""
        <div class="card card-error">
            <div style="font-size:0.85rem;font-weight:600;color:#ef4444;margin-bottom:6px;">
                ❌ Error
            </div>
            <div style="color:#fecaca;font-size:0.9rem;">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo / title
    st.markdown("""
    <div style="text-align:center;padding:8px 0 20px;">
        <div style="font-size:2.4rem;">🤖</div>
        <div style="font-size:1.05rem;font-weight:700;color:#f8fafc;letter-spacing:0.01em;">
            Support RAG
        </div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.40);margin-top:2px;">
            Multi-Agent · LangGraph · GPT-4o
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Status ──────────────────────────────────────────────
    st.markdown("#### 🔌 API Status")
    if st.button("Check Connection", key="btn_health", use_container_width=True):
        st.session_state.api_healthy = check_health()

    if st.session_state.api_healthy is True:
        st.markdown('<p class="dot-green">● API online</p>', unsafe_allow_html=True)
    elif st.session_state.api_healthy is False:
        st.markdown('<p class="dot-red">● API offline</p>', unsafe_allow_html=True)
        st.caption(f"Expected: `{API_BASE}`")
    else:
        st.markdown('<p class="dot-yellow">● Not checked</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Session ──────────────────────────────────────────────────
    st.markdown("#### 🔑 Session")
    st.code(st.session_state.session_id[:18] + "…", language=None)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New", use_container_width=True, key="btn_new_session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.session_state.hitl_draft = None
            st.rerun()
    with col_b:
        if st.button("Clear", use_container_width=True, key="btn_clear"):
            st.session_state.messages   = []
            st.session_state.hitl_draft = None
            st.rerun()

    st.markdown("---")

    # ── Session stats ─────────────────────────────────────────────
    st.markdown("#### 📊 Session Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Queries", st.session_state.stats["total"])
    c2.metric("HITL",    st.session_state.stats["hitl"])
    c3.metric("Errors",  st.session_state.stats["errors"])

    st.markdown("---")

    # ── Quick examples ────────────────────────────────────────────
    st.markdown("#### 💡 Quick Examples")
    examples = [
        "I need a refund for my broken headphones",
        "My app keeps crashing on iOS 17",
        "Why was I charged twice this month?",
        "What is your return policy?",
        "I want to speak to a manager",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["_quick_input"] = ex
            st.rerun()

    st.markdown("---")

    # ── Architecture ──────────────────────────────────────────────
    with st.expander("🗺 Architecture"):
        st.markdown("""
```
Query
  │
  ▼
IntentClassifier
  │
  ▼
HybridRetriever
(Dense + BM25)
  │
  ▼
DraftWriter (GPT-4o)
  │
  ▼
QA Agent ──► HITL ──┐
  │                  │
  ├── pass ──► Final │
  └── retry ◄────────┘
```
""")

    # ── Document ingestion ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Ingest Documents")
    source_dir = st.text_input("Source directory", value="data/docs", key="ingest_dir")
    if st.button("▶ Start Ingestion", use_container_width=True, key="btn_ingest"):
        code, body = api("POST", "/ingest", json={"source_dir": source_dir})
        if code == 202:
            st.session_state.ingest_job = body.get("job_id")
            st.success(f"Job accepted · `{st.session_state.ingest_job[:8]}…`")
        else:
            st.error(body.get("error", "Unknown error"))

    if st.session_state.ingest_job:
        if st.button("🔄 Poll Status", use_container_width=True, key="btn_poll"):
            code, body = api("GET", f"/ingest/status/{st.session_state.ingest_job}")
            status = body.get("status", "unknown")
            if status == "done":
                result = body.get("result", {})
                st.success(
                    f"✅ Done · {result.get('total_chunks', '?')} chunks · "
                    f"{result.get('duration_seconds', '?')}s"
                )
            elif status == "failed":
                st.error(f"❌ Failed: {body.get('error', '?')}")
            elif status == "running":
                st.info("⏳ Running…")
            else:
                st.warning(f"Status: `{status}`")


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="text-align:center;font-size:2rem;font-weight:700;
           background:linear-gradient(90deg,#818cf8,#34d399);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           margin-bottom:4px;">
    Customer Support AI
</h1>
<p style="text-align:center;color:rgba(255,255,255,0.45);font-size:0.9rem;margin-bottom:24px;">
    Powered by LangGraph · LlamaIndex · GPT-4o · ChromaDB
</p>
""", unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat", "🔍 About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Chat
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # ── Conversation history ──────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:48px 0;color:rgba(255,255,255,0.3);">
                <div style="font-size:3rem;margin-bottom:12px;">💬</div>
                <div style="font-size:1rem;">Ask anything about your order, bill, or product.</div>
                <div style="font-size:0.8rem;margin-top:8px;">Pick a quick example from the sidebar or type below.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                render_message(msg)

    st.markdown("---")

    # ── HITL resume panel ─────────────────────────────────────────
    if st.session_state.hitl_draft is not None:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
                    border-radius:14px;padding:16px 20px;margin-bottom:16px;">
            <div style="color:#f59e0b;font-weight:600;font-size:0.9rem;margin-bottom:8px;">
                ⏸ Human Review Required
            </div>
            <div style="color:rgba(255,255,255,0.6);font-size:0.82rem;">
                The QA agent flagged this draft for your review.
                Provide feedback and click <strong>Submit Feedback</strong>,
                or approve as-is by leaving it blank.
            </div>
        </div>
        """, unsafe_allow_html=True)

        feedback = st.text_area(
            "Your feedback (leave blank to approve as-is)",
            placeholder="e.g. 'Be more concise' · 'Add refund deadline' · Leave empty to approve",
            key="hitl_feedback_input",
            height=90,
        )
        col_fb1, col_fb2 = st.columns([3, 1])
        with col_fb1:
            if st.button("📤 Submit Feedback", type="primary", use_container_width=True, key="btn_submit_feedback"):
                with st.spinner("Resuming agent graph…"):
                    t0 = time.time()
                    code, body = api(
                        "POST", "/chat/resume",
                        json={"session_id": st.session_state.session_id, "human_feedback": feedback},
                    )
                    elapsed = time.time() - t0

                if code == 200:
                    st.session_state.hitl_draft = None
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": body.get("reply", ""),
                        "meta": {
                            "intent":      body.get("intent", ""),
                            "retry_count": body.get("retry_count", 0),
                            "latency_s":   elapsed,
                        },
                    })
                    st.session_state.stats["total"] += 1
                    st.rerun()
                elif code == 202:
                    # Another interrupt
                    st.session_state.hitl_draft = body.get("draft_reply", "")
                    st.session_state.messages.append({
                        "role": "hitl",
                        "content": st.session_state.hitl_draft,
                    })
                    st.session_state.stats["hitl"] += 1
                    st.rerun()
                else:
                    st.error(body.get("error", f"HTTP {code}"))
                    st.session_state.stats["errors"] += 1
        with col_fb2:
            if st.button("✕ Cancel", use_container_width=True, key="btn_cancel_hitl"):
                st.session_state.hitl_draft = None
                st.rerun()

    # ── Input bar ─────────────────────────────────────────────────
    else:
        # Check for quick-example injection
        default_val = st.session_state.pop("_quick_input", "")

        with st.form("chat_form", clear_on_submit=True):
            col_inp, col_send = st.columns([9, 1])
            with col_inp:
                user_input = st.text_input(
                    "Message",
                    value=default_val,
                    placeholder="Type your support query…",
                    label_visibility="collapsed",
                    key="chat_input",
                )
            with col_send:
                submitted = st.form_submit_button("Send ↑", use_container_width=True, type="primary")

        if submitted and user_input.strip():
            query = user_input.strip()

            # Add user message immediately
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()


# ── Deferred API call (runs after rerun if last message is user) ──────────────
if (st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and st.session_state.hitl_draft is None):

    query = st.session_state.messages[-1]["content"]

    with st.spinner("🤖 Thinking…"):
        t0 = time.time()
        code, body = api(
            "POST", "/chat",
            json={
                "query":      query,
                "session_id": st.session_state.session_id,
            },
        )
        elapsed = time.time() - t0

    if code == 200:
        st.session_state.messages.append({
            "role": "assistant",
            "content": body.get("reply", ""),
            "meta": {
                "intent":      body.get("intent", ""),
                "retry_count": body.get("retry_count", 0),
                "latency_s":   elapsed,
            },
        })
        st.session_state.stats["total"] += 1

    elif code == 202:
        draft = body.get("draft_reply", "")
        st.session_state.hitl_draft = draft
        st.session_state.messages.append({"role": "hitl", "content": draft})
        st.session_state.stats["hitl"] += 1

    elif code == 0:
        err = body.get("error", "Connection refused")
        st.session_state.api_healthy = False
        st.session_state.messages.append({"role": "error", "content": err})
        st.session_state.stats["errors"] += 1

    else:
        err = body.get("error", f"HTTP {code}")
        st.session_state.messages.append({"role": "error", "content": err})
        st.session_state.stats["errors"] += 1

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: About
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0;">🗺 System Architecture</h3>
            <p style="color:rgba(255,255,255,0.65);font-size:0.9rem;line-height:1.7;">
                A <strong>LangGraph cyclic state machine</strong> orchestrates four specialised agents:
            </p>
            <ol style="color:rgba(255,255,255,0.65);font-size:0.88rem;line-height:1.9;">
                <li><strong style="color:#818cf8;">IntentClassifier</strong> — Few-shot GPT-4o classifies the query intent</li>
                <li><strong style="color:#34d399;">HybridRetriever</strong> — Dense + BM25 fusion search over ChromaDB</li>
                <li><strong style="color:#f97316;">DraftWriter</strong> — Grounded reply generation with context injection</li>
                <li><strong style="color:#f472b6;">QA Agent</strong> — Structured verification (tone, accuracy, policy)</li>
            </ol>
            <p style="color:rgba(255,255,255,0.65);font-size:0.88rem;line-height:1.7;">
                Failed QA checks trigger a <strong>retry loop</strong>. After max retries or a low-confidence
                verdict, the graph pauses via <code>interrupt()</code> for <strong>human-in-the-loop</strong> review.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0;">🔌 API Endpoints</h3>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                    <td style="padding:6px 4px;color:#a5f3fc;"><code>GET /health</code></td>
                    <td style="padding:6px 8px;color:rgba(255,255,255,0.6);">Liveness probe</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                    <td style="padding:6px 4px;color:#a5f3fc;"><code>POST /chat</code></td>
                    <td style="padding:6px 8px;color:rgba(255,255,255,0.6);">Submit a support query</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                    <td style="padding:6px 4px;color:#a5f3fc;"><code>POST /chat/resume</code></td>
                    <td style="padding:6px 8px;color:rgba(255,255,255,0.6);">Resume after HITL review</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                    <td style="padding:6px 4px;color:#a5f3fc;"><code>POST /ingest</code></td>
                    <td style="padding:6px 8px;color:rgba(255,255,255,0.6);">Trigger document ingestion</td>
                </tr>
                <tr>
                    <td style="padding:6px 4px;color:#a5f3fc;"><code>GET /ingest/status/{job_id}</code></td>
                    <td style="padding:6px 8px;color:rgba(255,255,255,0.6);">Poll ingestion job</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0;">⚡ Tech Stack</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;">
                <div style="background:rgba(99,102,241,0.12);border-radius:10px;padding:12px;">
                    <div style="font-size:1.1rem;">🕸</div>
                    <div style="font-weight:600;color:#818cf8;font-size:0.85rem;">LangGraph</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">Cyclic agent graph + HITL</div>
                </div>
                <div style="background:rgba(16,185,129,0.12);border-radius:10px;padding:12px;">
                    <div style="font-size:1.1rem;">🦙</div>
                    <div style="font-weight:600;color:#34d399;font-size:0.85rem;">LlamaIndex</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">RAG + hybrid search</div>
                </div>
                <div style="background:rgba(249,115,22,0.12);border-radius:10px;padding:12px;">
                    <div style="font-size:1.1rem;">🤖</div>
                    <div style="font-weight:600;color:#fb923c;font-size:0.85rem;">GPT-4o</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">LLM backbone</div>
                </div>
                <div style="background:rgba(139,92,246,0.12);border-radius:10px;padding:12px;">
                    <div style="font-size:1.1rem;">🗄</div>
                    <div style="font-weight:600;color:#a78bfa;font-size:0.85rem;">ChromaDB</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">Vector store</div>
                </div>
                <div style="background:rgba(20,184,166,0.12);border-radius:10px;padding:12px;">
                    <div style="font-size:1.1rem;">⚡</div>
                    <div style="font-weight:600;color:#2dd4bf;font-size:0.85rem;">FastAPI</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">REST API layer</div>
                </div>
                <div style="background:rgba(244,63,94,0.12);border-radius:10px;padding:12px;">
                    <div style="font-size:1.1rem;">🎈</div>
                    <div style="font-weight:600;color:#fb7185;font-size:0.85rem;">Streamlit</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);">This demo UI</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0;">🎯 Intent Classes</h3>
        """, unsafe_allow_html=True)

        intent_descriptions = {
            "refund":    ("💸", "#f97316", "Money back / return requests"),
            "technical": ("🔧", "#3b82f6", "Product bugs & technical issues"),
            "billing":   ("🧾", "#8b5cf6", "Invoices, payments, subscriptions"),
            "general":   ("💬", "#10b981", "General enquiries & FAQs"),
            "escalate":  ("🚨", "#ef4444", "Sensitive complaints / manager request"),
        }

        for intent, (emoji, color, desc) in intent_descriptions.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:6px 0;
                        border-bottom:1px solid rgba(255,255,255,0.06);">
                <span class="badge" style="background:{color}22;color:{color};
                      border:1px solid {color}44;min-width:90px;text-align:center;">
                    {emoji} {intent}
                </span>
                <span style="color:rgba(255,255,255,0.55);font-size:0.85rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── How to run ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card" style="margin-top:12px;">
        <h3 style="margin-top:0;">🚀 How to Run</h3>
        <div style="font-size:0.88rem;color:rgba(255,255,255,0.65);line-height:1.8;">
            <strong>Terminal 1 — Start the backend:</strong>
        </div>
    """, unsafe_allow_html=True)
    st.code("uvicorn api.main:app --reload --port 8000", language="bash")
    st.markdown("""
    <div style="font-size:0.88rem;color:rgba(255,255,255,0.65);line-height:1.8;margin-top:12px;">
        <strong>Terminal 2 — Start this UI:</strong>
    </div>
    """, unsafe_allow_html=True)
    st.code("streamlit run ui/app.py", language="bash")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px;
            font-size:0.75rem;color:rgba(255,255,255,0.25);">
    Multi-Agent RAG Customer Support System · Sprint 4 Demo
</div>
""", unsafe_allow_html=True)
