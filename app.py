import streamlit as st
from pypdf import PdfReader
from core.llm import generate_answer
from rag.ingestion import split_text_into_chunks
from rag.embeddings import create_ephemeral_index, retrieve_relevant_chunks


st.set_page_config(page_title="ChatBot", page_icon="💬", layout="centered")

st.session_state.setdefault("username", "")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("name_submitted", False)
st.session_state.setdefault("show_uploader", False)
st.session_state.setdefault("collection", None)

if not st.session_state.name_submitted:
    st.session_state.page = "home"
else:
    st.session_state.page = "chat"

# ── Backend Logic ────────────────────────────────────────────────────────────────
def rag_pipeline(uploaded_pdf):
    if uploaded_pdf is not None:
        reader = PdfReader(uploaded_pdf)
        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        extracted_text = "\n".join(pages_text)
        chunks = split_text_into_chunks(extracted_text)
        collection = create_ephemeral_index(chunks)
        return collection
    else:
        return "No PDF uploaded."


# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Chat bubble container */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0 4px 12px 4px;
}

/* Each message row */
.msg-row {
    display: flex;
    width: 100%;
}
.msg-row.user  { justify-content: flex-end; }
.msg-row.bot   { justify-content: flex-start; }

/* Bubble itself */
.bubble {
    max-width: 72%;
    padding: 8px 12px;
    border-radius: 16px;
    font-size: 0.875rem;
    line-height: 1.45;
    word-wrap: break-word;
}

.bubble.user {
    background: #0b93f6;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}

.bubble.bot {
    background: #e9e9eb;
    color: #1c1c1e;
    border-bottom-left-radius: 4px;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .bubble.bot {
        background: #2c2c2e;
        color: #f2f2f7;
    }
}

/* Small Start Over button */
div[data-testid="stButton"].start-over-btn > button {
    font-size: 0.72rem;
    padding: 0.2rem 0.55rem;
    height: auto;
    min-height: unset;
    line-height: 1.2;
    border-radius: 6px;
    opacity: 0.55;
}
div[data-testid="stButton"].start-over-btn > button:hover { opacity: 1; }

/* Paperclip button */
div[data-testid="stButton"].attach-btn > button {
    font-size: 1.2rem;
    padding: 0.28rem 0.52rem;
    height: 2.6rem;
    border-radius: 8px;
    background: transparent;
    border: 1px solid rgba(128,128,128,0.3);
    line-height: 1;
}
div[data-testid="stButton"].attach-btn > button:hover {
    background: rgba(128,128,128,0.1);
}

/* Tighten file uploader */
div[data-testid="stFileUploader"] section {
    padding: 0.4rem 0.6rem;
    border-radius: 8px;
}
div[data-testid="stFileUploader"] label { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Helper: render all messages as HTML bubbles ──────────────────────────────
def render_messages(messages):
    html = '<div class="chat-wrapper">'
    for msg in messages:
        role_class = "user" if msg["role"] == "user" else "bot"
        content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += f'''
        <div class="msg-row {role_class}">
            <div class="bubble {role_class}">{content}</div>
        </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ── Home page ────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.session_state.username = st.text_input(
        "What should I call you?", placeholder="Enter your name"
    )
    if st.button("Start Chatting"):
        if st.session_state.username:
            st.session_state.username = st.session_state.username.strip()
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Hi {st.session_state.username}! 👋 How can I help you today?"
            })
            st.session_state.name_submitted = True
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")

# ── Chat page ────────────────────────────────────────────────────────────────
if st.session_state.page == "chat":
    st.title("💬 ChatBot")
    st.caption(f"Logged in as **{st.session_state.username}**")
    st.divider()

    render_messages(st.session_state.messages)

    # ── Input row: paperclip + chat input ───────────────────────────────────
    col_attach, col_input = st.columns([1, 11])

    with col_attach:
        st.markdown('<div class="attach-btn">', unsafe_allow_html=True)
        if st.button("📎", help="Attach a PDF"):
            st.session_state.show_uploader = not st.session_state.show_uploader
        st.markdown('</div>', unsafe_allow_html=True)

    with col_input:
        user_input = st.chat_input("Type your message here...")

    # ── PDF uploader (toggled by paperclip) ─────────────────────────────────
    if st.session_state.show_uploader:
        uploaded_pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
        if uploaded_pdf is not None:
            st.session_state.collection = rag_pipeline(uploaded_pdf)
            st.session_state.show_uploader = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📄 PDF \"{uploaded_pdf.name}\" I have read the content!"
            })
            st.rerun()

    # ── Handle user text input ───────────────────────────────────────────────
    if user_input and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": query})

        if st.session_state.collection:
            relevant_chunks = retrieve_relevant_chunks(query, st.session_state.collection)
            # Collect stream into full reply to keep bubble UI consistent
            with st.spinner("Thinking..."):
                reply = ""
                for chunk in generate_answer(query, relevant_chunks):
                    reply += chunk
        else:
            reply = "Please upload a PDF first."

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Small Start Over button ──────────────────────────────────────────────
    st.markdown('<div class="start-over-btn">', unsafe_allow_html=True)
    if st.button("↩ Start Over"):
        st.session_state.username = ""
        st.session_state.messages = []
        st.session_state.name_submitted = False
        st.session_state.show_uploader = False
        st.session_state.collection = None  # reset ephemeral index
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
