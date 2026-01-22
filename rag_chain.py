import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

INDEX_DIR = "embeddings/faiss_index"

# ✅ Detect Hugging Face Space
IS_HF = os.environ.get("SPACE_ID") is not None


def load_rag_chain():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ✅ Local LLM only (HF par disable)
    if IS_HF:
        llm = None
    else:
        llm = ChatOllama(
            model="mistral",
            temperature=0.0,
            timeout=120
        )

    def rag_answer(question: str, fund_filter: str = "All"):
        q = question.lower().strip()

        # ---- Greetings ----
        if q in ["hello", "hi", "hey", "assalam o alaikum", "salam"]:
            return (
                "Hello! 👋\n\n"
                "You can ask me about:\n"
                "- Islamic or Conventional fund performance\n"
                "- December 2025 Fund Manager Reports\n"
                "- Fund comparisons and summaries"
            ), []

        # ---- HF Demo Mode ----
        if llm is None:
            return (
                "⚠️ This is a Hugging Face UI demo.\n\n"
                "Full AI-powered answers are available when running the chatbot "
                "locally using a free open-source LLM (Ollama + Mistral)."
            ), []

        # ---- Retrieve documents ----
        docs = retriever.invoke(question)

        # Optional filter
        if fund_filter != "All":
            docs = [
                d for d in docs
                if fund_filter.lower() in d.page_content.lower()
            ]

        if not docs:
            return (
                "This information is not mentioned in the Fund Manager Reports."
            ), []

        # Limit context size for speed
        context = "\n\n".join(
            d.page_content[:900] for d in docs
        )

        prompt = (
            "You are a financial assistant.\n"
            "Answer ONLY using the context from Fund Manager Reports below.\n"
            "If the answer is not present, say it is not mentioned in the reports.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        response = llm.invoke(prompt)

        # Collect sources
        sources = list(
            {d.metadata.get("source", "Unknown source") for d in docs}
        )

        return response.content.strip(), sources

    return rag_answer
