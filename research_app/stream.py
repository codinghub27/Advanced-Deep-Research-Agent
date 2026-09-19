import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent; }
footer                     { display: none; }

/* ── Auth card ── */
.auth-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
}
.auth-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 16px;
    padding: 40px 36px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.auth-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
    text-align: center;
}
.auth-sub {
    font-size: 0.85rem;
    color: #6b7280;
    text-align: center;
    margin-bottom: 28px;
}
.tab-row {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
}
.tab-btn {
    flex: 1;
    padding: 8px 0;
    border-radius: 8px;
    border: 1px solid #2d3148;
    background: #0f1117;
    color: #6b7280;
    font-size: 0.9rem;
    cursor: pointer;
    text-align: center;
}
.tab-btn.active {
    background: #4f46e5;
    border-color: #4f46e5;
    color: #fff;
    font-weight: 600;
}

/* ── Inputs ── */
input[type="text"], input[type="password"] {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 8px !important;
}
input[type="text"]:focus, input[type="password"]:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.25) !important;
}
.stTextInput label { color: #9ca3af !important; font-size: 0.85rem !important; }

/* ── Primary button ── */
.stButton > button {
    width: 100%;
    background: #4f46e5;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 0;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover { background: #4338ca; }

/* ── Chat layout ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 0 16px 0;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 24px;
}
.chat-header-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.chat-header-title { font-size: 1.2rem; font-weight: 700; color: #fff; }
.chat-header-sub   { font-size: 0.78rem; color: #6b7280; }
.logout-btn {
    margin-left: auto;
    background: transparent;
    border: 1px solid #2d3148;
    color: #9ca3af;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    cursor: pointer;
}

/* ── Message bubbles ── */
.msg-row { display: flex; gap: 10px; margin-bottom: 20px; align-items: flex-start; }
.msg-row.user  { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0; margin-top: 2px;
}
.avatar.user { background: linear-gradient(135deg, #4f46e5, #7c3aed); }
.avatar.bot  { background: linear-gradient(135deg, #0ea5e9, #6366f1); }
.bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #e2e8f0;
}
.bubble.user {
    background: #1e2130;
    border-top-right-radius: 4px;
}
.bubble.bot {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-top-left-radius: 4px;
}

/* ── Progress steps ── */
.progress-wrap {
    display: flex; flex-direction: column; gap: 6px;
    margin-bottom: 10px;
}
.progress-step {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.8rem; color: #6b7280;
    animation: fadeIn 0.3s ease;
}
.progress-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #4f46e5; flex-shrink: 0;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }

/* ── Chat input bar ── */
.stChatInputContainer, [data-testid="stChatInput"] {
    background: #1a1d27 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
}

/* ── Error / success alerts ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "token": None,
        "username": None,
        "messages": [],          # [{"role": "user"|"assistant", "content": str}]
        "auth_tab": "login",     # "login" | "register"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
#  API helpers
# ─────────────────────────────────────────────
def api_register(username: str, password: str):
    try:
        r = requests.post(f"{API_BASE}/auth/register",
                          json={"username": username, "password": password}, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"detail": str(e)}


def api_login(username: str, password: str):
    try:
        r = requests.post(f"{API_BASE}/auth/login",
                          json={"username": username, "password": password}, timeout=10)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"detail": str(e)}


def stream_research(question: str, token: str):
    """Generator that yields (type, value) tuples from the SSE stream."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"question": question}
    try:
        with requests.post(
            f"{API_BASE}/api/research/stream",
            json=payload,
            headers=headers,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                yield data
    except requests.exceptions.RequestException as e:
        yield {"type": "error", "content": str(e)}


# ─────────────────────────────────────────────
#  Auth page
# ─────────────────────────────────────────────
def render_auth():
    # Centre the card
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="auth-title">🔬 Research Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">AI-powered deep research at your fingertips</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        # ── Login ──
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="your username")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In")

            if submitted:
                if not username or not password:
                    st.error("Please fill in both fields.")
                else:
                    with st.spinner("Signing in…"):
                        code, resp = api_login(username, password)
                    if code == 200:
                        st.session_state.token = resp["access_token"]
                        st.session_state.username = username
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(resp.get("detail", "Login failed."))

        # ── Register ──
        with tab_register:
            with st.form("register_form", clear_on_submit=True):
                new_user = st.text_input("Username", placeholder="choose a username", key="reg_user")
                new_pass = st.text_input("Password", type="password", placeholder="••••••••", key="reg_pass")
                new_pass2 = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="reg_pass2")
                reg_submitted = st.form_submit_button("Create Account")

            if reg_submitted:
                if not new_user or not new_pass:
                    st.error("Please fill in all fields.")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Creating account…"):
                        code, resp = api_register(new_user, new_pass)
                    if code == 200:
                        st.success("Account created! Sign in to continue.")
                    else:
                        st.error(resp.get("detail", "Registration failed."))


# ─────────────────────────────────────────────
#  Chat page
# ─────────────────────────────────────────────
def render_chat():
    # ── Header ──
    header_col, logout_col = st.columns([5, 1])
    with header_col:
        st.markdown(f"""
        <div class="chat-header">
            <div class="chat-header-icon">🔬</div>
            <div>
                <div class="chat-header-title">Research Agent</div>
                <div class="chat-header-sub">Signed in as <b>{st.session_state.username}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with logout_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign out", key="logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()

    # ── Message history ──
    for msg in st.session_state.messages:
        _render_bubble(msg["role"], msg["content"])

    # ── Chat input ──
    question = st.chat_input("Ask anything to research…")
    if not question:
        return

    # Show user bubble immediately
    st.session_state.messages.append({"role": "user", "content": question})
    _render_bubble("user", question)

    # ── Streaming response ──
    with st.chat_message("assistant", avatar="🤖"):
        progress_placeholder = st.empty()
        answer_placeholder   = st.empty()

        progress_steps = []
        answer_tokens  = []
        in_think       = False
        think_buffer   = ""

        for event in stream_research(question, st.session_state.token):
            etype = event.get("type")

            if etype == "progress":
                label = event.get("label", event.get("node", ""))
                progress_steps.append(label)
                steps_html = "".join(
                    f'<div class="progress-step"><div class="progress-dot"></div>{s}</div>'
                    for s in progress_steps
                )
                progress_placeholder.markdown(
                    f'<div class="progress-wrap">{steps_html}</div>',
                    unsafe_allow_html=True,
                )

            elif etype == "token":
                raw = event.get("content", "")

                # Filter out <think>…</think> tokens mid-stream
                think_buffer += raw
                while True:
                    if in_think:
                        end = think_buffer.find("</think>")
                        if end != -1:
                            think_buffer = think_buffer[end + len("</think>"):]
                            in_think = False
                        else:
                            think_buffer = ""
                            break
                    else:
                        start = think_buffer.find("<think>")
                        if start != -1:
                            clean_part = think_buffer[:start]
                            if clean_part:
                                answer_tokens.append(clean_part)
                            think_buffer = think_buffer[start + len("<think>"):]
                            in_think = True
                        else:
                            answer_tokens.append(think_buffer)
                            think_buffer = ""
                            break

                answer_placeholder.markdown(
                    f'<div class="bubble bot">{"".join(answer_tokens)}</div>',
                    unsafe_allow_html=True,
                )

            elif etype == "done":
                progress_placeholder.empty()

            elif etype == "error":
                st.error(f"Stream error: {event.get('content')}")

        final_answer = "".join(answer_tokens).strip()
        if final_answer:
            st.session_state.messages.append({"role": "assistant", "content": final_answer})


def _render_bubble(role: str, content: str):
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user">👤</div>
            <div class="bubble user">{content}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar bot">🤖</div>
            <div class="bubble bot">{content}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────
if st.session_state.token:
    render_chat()
else:
    render_auth()
