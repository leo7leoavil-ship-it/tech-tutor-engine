---
title: Tech Tutor Engine
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Tech Tutor Engine (MaaS Backend)

Asynchronous **Model-as-a-Service** for *Asistente de Inteligencia Artificial Especializado en Inglés Técnico para Ingenieros del Área de Tecnología*.

There is **no frontend** in this repository. Clients (mobile, future React/Next.js work, or internal tools) call REST and WebSocket APIs.

| Layer | Implementation |
| --- | --- |
| API | FastAPI + Uvicorn, Pydantic v2 |
| ASR | Whisper Large-v3-Turbo via `faster-whisper` (in-memory WebM/MP3) |
| RAG | Existing ChromaDB collection `tech_tutor_corpus` + `all-MiniLM-L6-v2` (384-d) |
| LLM | Llama 3 through an OpenAI-compatible API (Groq by default) + LangChain structured JSON |
| Deploy | Multi-stage `Dockerfile`, Hugging Face Spaces port **7860** |

The vector store is **not** rebuilt here. Copy the Data Track persist directory into `data/technical_corpus_db`.

## OpenAPI surface

Interactive docs: `/docs` (Swagger UI) and `/redoc`. Schema: `/openapi.json`.

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | none | Readiness: Chroma bound, Whisper loaded, LLM key present |
| `GET` | `/api/v1/health` | none | Same probe under the versioned prefix |
| `POST` | `/api/v1/query` | Bearer | Text-only RAG retrieval (`top_k` cosine neighbours) |
| `POST` | `/api/v1/evaluate/voice` | Bearer | Multipart audio → ASR → RAG → Llama 3 `EvaluationResponse` |
| `WS` | `/api/v1/ws/practice?token=` | query token | Binary audio frames + JSON `evaluate` control |

### `POST /api/v1/query`

```json
{
  "query_text": "Explain Kubernetes rolling updates.",
  "top_k": 3
}
```

### `POST /api/v1/evaluate/voice`

`multipart/form-data`:

- `audio_file`: `.webm` or `.mp3`
- `target_scenario`: default `system_design_interview`

Response (`EvaluationResponse`):

```json
{
  "status": "success",
  "execution_time_ms": 1842.17,
  "asr_metrics": {
    "transcription": "...",
    "detected_language": "en",
    "confidence_score": 0.91
  },
  "evaluation_core": {
    "overall_score": 82.0,
    "lexical_density_ratio": 0.61,
    "grammar_accuracy_score": 88.0,
    "detected_technical_terms": ["Kubernetes", "rolling update"]
  },
  "feedback_details": [
    {
      "type": "lexicon",
      "original_phrase": "...",
      "suggested_phrase": "...",
      "rag_source_context": "...",
      "explanation": "..."
    }
  ]
}
```

### WebSocket practice protocol

1. Connect to `/api/v1/ws/practice?token=<API_KEY>`.
2. Send binary frames (WebM/MP3 chunks). Server ACKs `buffered_bytes`.
3. Send text: `{"event":"evaluate","target_scenario":"system_design_interview"}`.
4. Receive `{"event":"evaluation","payload": { ... EvaluationResponse } }`.
5. Optional: `{"event":"reset"}`, `{"event":"ping"}`.

## Local execution

Python 3.11+, FFmpeg on `PATH` (required by PyAV / faster-whisper).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # set API_KEY and LLM_API_KEY
```

Place the pre-built ChromaDB files under `data/technical_corpus_db` (collection name **must** remain `tech_tutor_corpus`).

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Spaces / production container:

```bash
docker build -t tech-tutor-engine .
docker run --rm -p 7860:7860 --env-file .env tech-tutor-engine
```

Hugging Face Spaces secrets: set `API_KEY` and `LLM_API_KEY` (never bake keys into the image).

## Tests

```bash
pytest
```

Unit tests mock Whisper, Chroma, and the LLM. They set `SKIP_HEAVY_INIT=true` so weights are not downloaded.

## CI/CD

`.github/workflows/deploy.yml` force-syncs `main` to a Docker Space. Configure repository secrets:

- `HF_TOKEN` — write token
- `HF_USERNAME` — Space namespace
- `HF_SPACE` — Space name

If `data/technical_corpus_db` exceeds ~100MB, Git LFS is already declared in `.gitattributes`.
