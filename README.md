# RAG FMR Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about Fund Manager Reports (FMRs), including both Islamic and Conventional funds.

## Features
- Conversational Q&A over Fund Manager Reports
- PDF and website-based ingestion
- Islamic and Conventional fund filtering
- Source-grounded answers
- Extendable for historical reports
- Uses free open-source LLMs (local deployment)

## Tech Stack
- Python
- Streamlit
- FAISS
- Sentence Transformers
- Ollama (Mistral)

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
