# DraftMate – Legal Drafting Studio

An AI-assisted drafting workspace for Indian legal work. It drafts documents in the style of a reference file, **checks every citation it can**, finds supporting and adverse cases from a case file you control, reviews documents for gaps, translates them, and takes voice instructions.

Built for **Kodekurrent Season 2** (RGIPT, Amethi) — problem statement: *Legal Drafting Studio: An AI-Powered Legal Drafting, Verification and Voice Agent*.

> DraftMate assists a lawyer. It does not replace professional judgment. Review all output before filing.

## Features

| Area | What it does |
|---|---|
| Drafting | Skills library (notices, contracts, petitions, NDAs, replies and more), templates with `[placeholders]`, prompt-strength meter and prompt improver |
| Reference style | Upload a PDF/DOCX/TXT; a style profile is learned and applied to new drafts |
| Context management | The reference is split into chunks and only the most relevant ones (BM25) are sent to the model; the UI shows tokens saved |
| Citation verification | Finds case and statute citations and labels them verified / partial / unverified. Checks the case name against the citation number, offers a one-click correction, and looks up Indian Kanoon |
| Similar cases | Supporting and **adverse** authorities from `cases.json`, with "why it matched" and a draft-a-response action |
| Correction Review | Upload a PDF, PPTX, DOCX or image: what is present, what is missing, what to fix |
| Translate a document | PDF, PPTX, DOCX, TXT or image into 12 Indian languages, structure preserved |
| Voice | Whisper (faster-whisper) transcription, or the browser's speech recognition as a fallback. Overwrites and deletes ask for confirmation, and Undo restores the previous version |
| Models | Any OpenAI-compatible provider (Groq, OpenAI, Gemini, Anthropic, OpenRouter, Ollama, custom) with your own API key |
| Export | Word (.docx) with headings and bold |
| Interface | Light and dark themes, saved draft history in the browser |

## Run it

Requires **Python 3.12** (newer versions may lack packages for Whisper and OCR).

```bash
python -m venv venv312
venv312\Scripts\activate          # Windows   (Mac/Linux: source venv312/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Mac/Linux: cp .env.example .env
python -m uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000 in Edge or Chrome.

**Connect an AI:** put a key in `.env` (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), or use **AI settings** in the app. With no key the app runs in demo mode and shows sample output only. Never commit `.env`.

## How it works

```
Browser (static/index.html)  ──►  FastAPI (main.py)
   editor, panels, voice           ├─ llm(): any OpenAI-compatible provider
                                   ├─ BM25 retrieval over the uploaded reference
                                   ├─ citation checker: regex + authorities.json + Indian Kanoon lookup
                                   ├─ case search over cases.json (supporting / adverse)
                                   ├─ Whisper transcription, OCR, PPTX/PDF/DOCX readers
                                   └─ python-docx export
```

Data files you can edit without code: `skills.json` (skill prompts), `authorities.json` (citations the verifier trusts), `cases.json` (similar and adverse cases).

## Honest limitations

- The verifier trusts a **small local list** plus a best-effort Indian Kanoon lookup. It does not cover all Indian case law, and "verified" means "matches our list".
- The entries in `cases.json` are starter samples and must be reviewed against the original judgments before real use (each shows an "unreviewed" badge until marked reviewed).
- Similar-case search is keyword based (BM25), not semantic, and it only searches the case file.
- Reviews and translations are AI output and can be wrong. OCR works best on clear English print. Scanned PDFs must be uploaded as images.
- Draft history is stored in your browser only.

## Reference tools

*Edit this section so it matches what your team actually did.*

- **Whisper:** used through `faster-whisper` for voice input.
- **Context reduction:** our own BM25 retrieval over document chunks.
- **Skills framework:** our own, in `skills.json`.
- **Not integrated:** VoxCPM2 (no spoken replies yet) and Headroom.
- **MarkItDown:** used when installed; otherwise a built-in fallback reads PDF, DOCX and PPTX.

## Team

**Team Code-Oligarchs**

- Nishant Ranjan
- Dipender
- Vishav Pratap Singh
- Sampreeti Pramanik
