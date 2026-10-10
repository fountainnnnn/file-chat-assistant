# File Chat Assistant

Ask questions about a document and get answers that cite the page they came from. Works with PDFs, Word files, text files, and scanned pages or photos of documents.

**Live demo:** https://ngyuhang.com/file-chat-assistant/

## How it works

- **Loads the document** and keeps page numbers attached to every chunk of text.
- **Reads scanned pages and images with a vision model** when a page has little or no text layer.
- **Answers with retrieval:** the most relevant chunks go to the LLM, which must cite them like `[p. 3]`. The response includes a citations list (page and snippet), shown under each answer.

## Keys

`DEEPSEEK_API_KEY` for answers, plus one of `VISION_API_KEY`, `OPENROUTER_API_KEY` or `OPENAI_API_KEY` for scanned pages (optional).

## Run locally

```bash
# API
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...   # plus the optional keys listed above
CORS_ORIGINS=http://localhost:5173 uvicorn app:app --port 7860

# Frontend (second terminal)
cd frontend
python -m http.server 5173
```

Open http://localhost:5173. The page calls the API at `http://localhost:7860` by default; add `?api=<url>` to point it elsewhere.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt 2>/dev/null || pip install pytest httpx
pytest
```

All model calls are mocked in the tests, so they run offline and cost nothing.

See [`backend/README.md`](backend/README.md) for every endpoint and environment variable.

## Known limitations

- Sessions live in memory and expire after an hour, so restarting the server clears them.
- Word and text files have no pages, so they're cited by part instead.

## Project layout

- `backend/`: FastAPI service (the same code that runs the live demo)
- `frontend/`: static page (HTML, CSS, JavaScript)

## License

MIT
