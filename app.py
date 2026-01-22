import streamlit as st
from rag_chain import load_rag_chain
import os
import shutil

st.set_page_config(
    page_title="FMR AI Chatbot",
    layout="wide"
)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")

fund_filter = st.sidebar.radio(
    "Fund Type",
    ["All", "Islamic", "Conventional"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📄 Upload Historical FMR")
uploaded_file = st.sidebar.file_uploader(
    "Upload FMR PDF",
    type=["pdf"]
)

if uploaded_file:
    save_path = os.path.join("data/pdfs", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("PDF uploaded. Rebuild index to include it.")

st.sidebar.markdown("---")
st.sidebar.info(
    "This chatbot uses Retrieval-Augmented Generation (RAG)\n"
    "over official Fund Manager Reports."
)

# ---------- Main UI ----------
st.title("💬 Fund Manager Report AI Chatbot")
st.caption(
    "Ask questions about NBP Fund Manager Reports "
    "(Islamic & Conventional)."
)

@st.cache_resource
def init_rag():
    return load_rag_chain()

rag_answer = init_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_query = st.chat_input(
    "Ask about fund performance, returns, comparisons..."
)

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag_answer(user_query, fund_filter)
            st.markdown(answer)

            if sources:
                st.markdown("**Sources:**")
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
