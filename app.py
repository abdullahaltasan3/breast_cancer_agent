"""Chat web interface for the breast cancer data agent (Step 5).

Run:  streamlit run app.py
The interface is the link between the user and the agent: every question goes to
agent.ask(), which works on data/final_dataset.csv. Settings are read from .env
(OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL) and can be changed under Settings.
"""
import os
import re
import time
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Data Agent", page_icon="🔬", layout="centered",
                   initial_sidebar_state="expanded")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import graph as agent_graph  # noqa: E402
from agent.tools import load_data  # noqa: E402

ASSISTANT_ICON = ":material/auto_awesome:"
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "0") == "1"  # set to 1 on Render
DISCLAIMER = "Educational project. Age and gender are simulated. Not for medical use."
SUGGESTIONS = [
    ("🎗️", "How many people have a malignant tumour?"),
    ("👥", "How many females and how many males are there?"),
    ("🩺", "I have a small tumour, is it dangerous?"),
    ("⚖️", "Compare the average radius of benign and malignant cases."),
    ("📈", "Show the distribution of area for each diagnosis."),
    ("🔗", "Which variables are most strongly associated with the diagnosis?"),
]
TOOL_LABELS = {
    "dataset_overview": "Reading the dataset overview",
    "describe_variable": "Calculating statistics",
    "count_cases": "Counting cases",
    "compare_groups": "Comparing benign and malignant cases",
    "correlation": "Calculating correlations",
    "find_outliers": "Finding outliers",
    "make_chart": "Drawing a chart",
    "predict_case": "Estimating a new case",
}

# ---------------------------------------------------------------- styles ----
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 780px;}
  h1, h2, h3, .hero-title {font-family: Georgia, 'Times New Roman', serif; font-weight: 500;}
  .hero {text-align: center; margin: 7vh 0 1.4rem 0;}
  .orb {width: 92px; height: 92px; margin: 0 auto 1.1rem auto; border-radius: 50%;
        background: conic-gradient(from 0deg, #ff6b9d, #c56cf0, #5f8dff, #2ed3c6, #ffb86b, #ff6b9d);
        filter: blur(1px) saturate(1.15);
        box-shadow: 0 0 38px 8px rgba(197, 108, 240, .35), inset 0 0 22px rgba(255,255,255,.55);
        animation: spin 6s linear infinite, breathe 2.8s ease-in-out infinite;}
  .orb::after {content: ""; display: block; width: 100%; height: 100%; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, rgba(255,255,255,.85), rgba(255,255,255,0) 55%);}
  @keyframes spin {to {transform: rotate(360deg);}}
  @keyframes breathe {0%, 100% {scale: 1;} 50% {scale: 1.08;}}
  .greet {font-family: Georgia, serif; font-size: 2.4rem; line-height: 1.2; color: #1F1E1D; margin: 0;}
  [data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li, .explain {
      unicode-bidi: plaintext; text-align: start;}
  .stButton > button > div {justify-content: flex-start; width: 100%;}
  .intro {font-size: 1.05rem; color: #3D3C38; margin: .5rem auto .2rem auto; max-width: 560px;
          overflow: hidden; white-space: nowrap; border-right: 2px solid #C96442; width: 0;
          animation: typing 2.6s steps(60, end) .4s forwards, caret .8s step-end 4,
                     hidecaret 0s 3.4s forwards;}
  @keyframes hidecaret {to {border-right-color: transparent;}}
  @keyframes typing {to {width: 100%;}}
  @keyframes caret {50% {border-color: transparent;}}
  .langs {color: #8A8880; font-size: .85rem; margin-top: .3rem;}
  .hero-icon {font-size: 2.2rem; color: #C96442;}
  .hero-title {font-size: 2.2rem; color: #1F1E1D; margin: .2rem 0 .4rem 0;}
  .hero-sub {color: #6B6A65; font-size: 1rem;}
  .disclaimer {text-align: center; color: #8A8880; font-size: .78rem; margin-top: .6rem;}
  .stButton > button {border-radius: 12px; border: 1px solid #E3E0D6; background: #FFFFFF;
                      color: #3D3C38; text-align: left; justify-content: flex-start;
                      padding: .7rem .9rem; font-size: .9rem;}
  .stButton > button p {white-space: normal; text-align: left;}
  [data-testid="stChatMessageAvatarUser"] {display: none;}
  [data-testid="stChatMessage"] [data-testid="stIconMaterial"] {color: #C96442;}
  .stButton > button:hover {border-color: #C96442; color: #C96442; background: #FFFFFF;}
  [data-testid="stChatMessage"] {background: transparent; padding: .4rem 0;}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
      flex-direction: row-reverse; text-align: left;}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
      [data-testid="stChatMessageContent"] {
      background: #F0EEE6; border-radius: 16px; padding: .55rem .95rem;
      max-width: 85%; flex-grow: 0; margin-left: auto;}
  [data-testid="stChatInput"] {border-radius: 18px;}
  [data-testid="stSidebar"] {background: #F0EEE6;}
  .side-title {font-family: Georgia, serif; font-size: 1.15rem; color: #1F1E1D; margin-bottom: .3rem;}
  .conn {font-size: .8rem; color: #3D7A4E; background: #E7F1E4; border-radius: 8px;
         padding: .45rem .6rem; margin: .3rem 0 .8rem 0;}
  .conn.off {color: #9B3B2B; background: #F7E3DE;}
  .explain {background: #FFFFFF; border: 1px solid #E8E5DA; border-left: 3px solid #C96442;
            border-radius: 10px; padding: .7rem .9rem; color: #3D3C38; margin: .6rem 0;}
  .note {color: #8A8880; font-size: .8rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- state -----
def new_chat():
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.current = cid


st.session_state.setdefault("chats", {})
if not st.session_state.chats:
    new_chat()
chat = st.session_state.chats[st.session_state.current]


@st.cache_data(show_spinner=False)
def dataset_status():
    try:
        df = load_data()
        return True, f"{len(df)} records × {df.shape[1]} columns"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown('<div class="side-title">🔬 Breast Cancer Data Agent</div>', unsafe_allow_html=True)
    ok, info = dataset_status()
    st.markdown(f'<div class="conn{"" if ok else " off"}">{"● Connected to" if ok else "● Not connected:"} '
                f'final_dataset.csv<br>{info}</div>', unsafe_allow_html=True)
    if st.button("＋  New chat", use_container_width=True, key="new_chat"):
        new_chat()
        st.rerun()
    st.caption("Chats")
    for cid, c in reversed(list(st.session_state.chats.items())):
        label = ("▸ " if cid == st.session_state.current else "") + c["title"]
        if st.button(label, key=f"chat_{cid}", use_container_width=True):
            st.session_state.current = cid
            st.rerun()
    st.divider()
    # On a public deployment (PUBLIC_MODE=1) visitors must not change the server's key,
    # model or provider address: these come only from the host's environment variables.
    if PUBLIC_MODE:
        show_steps = st.toggle("Show agent steps", value=True)
        model, base_url = os.getenv("LLM_MODEL", "gpt-5.4-mini"), os.getenv("OPENAI_BASE_URL", "")
    else:
        with st.expander("⚙️ Settings"):
            key = st.text_input("API key", value="", type="password",
                                help="Leave empty to use OPENAI_API_KEY from .env")
            model = st.text_input("Model", value=os.getenv("LLM_MODEL", "gpt-5.4-mini"))
            base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", ""),
                                     help="e.g. https://api.groq.com/openai/v1")
            show_steps = st.toggle("Show agent steps", value=True)
        if key:
            os.environ["OPENAI_API_KEY"] = key
        os.environ["LLM_MODEL"] = model
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        else:
            os.environ.pop("OPENAI_BASE_URL", None)
    st.caption(DISCLAIMER)

settings = (os.getenv("OPENAI_API_KEY", ""), model, base_url)
if st.session_state.get("settings") != settings:
    st.session_state.settings = settings
    agent_graph._GRAPH = None  # rebuild the agent with the new model settings


# ---------------------------------------------------------------- render ----
def steps_label(out: dict) -> str:
    tools = out.get("tools_used", [])
    if any(e["event"] == "refuse_personal_medical_request" for e in out.get("trace", [])):
        return "Safety check · personal medical question, no advice given"
    revised = [e["event"] for e in out.get("trace", [])].count("revise")
    if not tools:
        return "Answered without the data tools" if out.get("trace") else "No answer"
    label = "Answered from the dataset · " + ", ".join(dict.fromkeys(t["name"] for t in tools))
    return label + (f" · checked and revised {revised}×" if revised else " · checked ✓")


def render_steps(out: dict) -> None:
    with st.expander(f"🔧 {steps_label(out)}"):
        for e in out.get("trace", []):
            detail = {k: v for k, v in e.items() if k not in {"t", "agent", "event"}}
            st.markdown(f"**{e['agent']}** · `{e['event']}`")
            if detail:
                st.json(detail, expanded=False)


def render_extras(out: dict) -> None:
    if out.get("explanation"):
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out["explanation"])  # **bold** inside the HTML box
        st.markdown(f'<div class="explain">💡 {text}</div>', unsafe_allow_html=True)
    for t in out.get("tables", []):
        st.caption(t["title"])
        st.dataframe(pd.DataFrame(t["rows"]), use_container_width=True, hide_index=True)
    for c in out.get("charts", []):
        if os.path.exists(c):
            st.image(c)
    for n in out.get("notes", []):
        if n != agent_graph.DISCLAIMER:
            st.markdown(f'<div class="note">ℹ️ {n}</div>', unsafe_allow_html=True)


def render_assistant(out: dict, steps: bool) -> None:
    if steps and out.get("trace"):
        render_steps(out)
    st.markdown(out["answer"])
    render_extras(out)


def stream_words(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.012)


# ---------------------------------------------------------------- page ------
question = st.chat_input("Ask me anything about the data… / اسألني عن البيانات…")
question = question or st.session_state.pop("pending", None)

if not chat["messages"] and not question:
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.markdown(f"""
      <div class="hero">
        <div class="orb"></div>
        <div class="greet">{greeting} 👋</div>
        <div class="intro">I'm your breast cancer data assistant. What would you like to know?</div>
        <div class="langs">I answer from 569 real cases · English or العربية</div>
      </div>""", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, text) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{icon}  {text}", key=f"ex{i}", use_container_width=True):
            st.session_state.pending = text
            st.rerun()
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

for m in chat["messages"]:
    with st.chat_message(m["role"], avatar=None if m["role"] == "user" else ASSISTANT_ICON):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_assistant(m["out"], show_steps)

if question:
    if not chat["messages"]:
        chat["title"] = question[:32] + ("…" if len(question) > 32 else "")
    history = [{"question": u["content"], "answer": a["out"]["answer"]}
               for u, a in zip(chat["messages"][0::2], chat["messages"][1::2])]
    chat["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        status = st.status("Thinking…", expanded=False)
        try:
            out = agent_graph.ask(question, history=history)
        except Exception as e:  # show the problem instead of crashing the page
            out = {"answer": f"**Something went wrong:** {e}", "explanation": "", "notes": [],
                   "tables": [], "charts": [], "trace": [], "tools_used": []}
        with status:
            for t in out.get("tools_used", []):
                st.write(f"✓ {TOOL_LABELS.get(t['name'], t['name'])} (`{t['name']}`)")
            if show_steps:
                for e in out.get("trace", []):
                    st.caption(f"{e['agent']} · {e['event']}")
        failed = not out.get("trace")
        status.update(label=steps_label(out), state="error" if failed else "complete", expanded=False)
        st.write_stream(stream_words(out["answer"]))
        render_extras(out)
    chat["messages"].append({"role": "assistant", "content": out["answer"], "out": out})
