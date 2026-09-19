# Smart Chat Assistant

A production-style Streamlit app that turns uploaded files into a grounded file intelligence assistant. It supports PDF, DOCX, TXT, CSV, Excel, JSON, smart suggestions, structured analytics, SQL/Excel helper output, persistent chat history, and exports to PDF, DOCX, or TXT.

## Features

- Upload multiple documents: PDF, DOCX, TXT, CSV, XLSX/XLS, JSON
- FAISS vector search with SentenceTransformers embeddings
- Grounded file-only answers from retrieved chunks
- LLM-backed reasoning using Groq or Gemini when configured
- General AI chat without file upload when Groq, OpenAI, or Gemini is configured
- Structured table analytics for CSV, Excel, and tabular JSON
- SQL query and Excel formula helpers for analytical answers
- Dashboard, History, and Settings pages
- 10 suggested questions after indexing
- Streamlit chat history
- Export filters:
  - File Q&A only
  - Explanation only
  - Full conversation
- Export formats:
  - PDF
  - DOCX
  - TXT

## Project Structure

```text
rag bot/
├── app/
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── services/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── exporter.py
│   ├── file_loader.py
│   ├── llm.py
│   ├── models.py
│   ├── rag.py
│   └── utils.py
├── data/
│   ├── history/
│   ├── uploads/
│   └── vectorstores/
├── exports/
├── prompts/
│   └── example_prompts.md
├── .streamlit/
│   └── config.toml
├── .env.example
├── requirements.txt
├── app.py
└── streamlit_app.py
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Set one provider in `.env`:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

or:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

or:

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-2.5-flash
```

The app also has a local extractive fallback, so it can run without an API key for basic grounded answers and demo suggestions.

## Run

```bash
streamlit run app.py --server.fileWatcherType none
```

Then open the local URL Streamlit prints, usually:

```text
https://aymcp4actmsnw3vevtugts.streamlit.app/
```

## How It Works

1. Files are loaded into text and structured table metadata where available.
2. Text is split into retrieval chunks with overlap.
3. Chunks are embedded with SentenceTransformers.
4. FAISS stores vectors in memory for fast retrieval.
5. CSV, Excel, and tabular JSON questions are answered by deterministic Pandas-style table logic first.
6. Document questions retrieve the top relevant chunks from the active file set.
7. If a question is unrelated to files, the active LLM provider answers normally.
8. Provider calls for file answers receive only retrieved context and the user question.
9. Conversation exports include timestamps, questions, answers, explanations, and citations.

## Notes for Production Deployment

- Put API keys in environment variables or Streamlit secrets, never in source code.
- For many users or very large documents, persist FAISS indexes per user/session and add background processing.
- Add authentication before deploying publicly.
- Consider object storage for uploads and a managed vector DB for multi-tenant SaaS use.
- Review provider model names periodically because LLM model availability changes over time.
