"""DraftMate backend (FastAPI). Run: uvicorn main:app --reload"""
import os, io, re, json, tempfile
from pathlib import Path
from urllib.parse import quote
import requests
from dotenv import load_dotenv
import time, uuid
from collections import defaultdict, deque
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from rank_bm25 import BM25Okapi            # picks only relevant chunks
from docx import Document                  # real Word export

load_dotenv()
BASE = Path(__file__).parent
KEY, MODEL = os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
client = OpenAI(api_key=KEY or "none", base_url=os.getenv("LLM_BASE_URL"))
SKILLS = json.loads((BASE / "skills.json").read_text(encoding="utf-8"))
AUTH = json.loads((BASE / "authorities.json").read_text(encoding="utf-8"))
RULES = "\nNever invent case citations. Cite only authorities you are certain exist; otherwise write [UNVERIFIED]. Output in Markdown: put the title on its own line as '# TITLE'; use **bold** for the Subject line, party names, defined terms and clause headings; no other Markdown."
STORE = {"chunks": [], "bm25": None, "raw": 0, "style": ""}   # the uploaded reference lives here
_whisper = None
app = FastAPI()

DEMO = """# LEGAL NOTICE

**To:** [Tenant Name], [Address]

**Subject:** Notice for non-payment of rent

1. That you are a tenant of my client and have defaulted on rent for three months.
2. That notice is served under Section 106 of the Transfer of Property Act, 1882.
3. As held in Maneka Gandhi v. Union of India, AIR 1973 SC 1461, due process must be followed.
4. See also Puttaswamy v. Union of India, (2017) 10 SCC 1 on dignity.
5. In Ramesh v. Suresh, AIR 2019 SC 9999, default of rent was held a ground for eviction.

You must pay the arrears within 15 days, failing which legal action will follow.

[Advocate Name]"""

def tok(s): return re.findall(r"\w+", s.lower())

PROV = {}   # provider name -> {"client": OpenAI(...), "models": [...]}  (kept in memory)

def add_provider(name, base_url, key, models):
    PROV[name] = {"client": OpenAI(api_key=key or "none", base_url=base_url or None), "models": models}

if KEY: add_provider("default", os.getenv("LLM_BASE_URL"), KEY, [m.strip() for m in os.getenv("LLM_MODELS", MODEL).split(",") if m.strip()])

def llm(system, user, demo="[demo mode: add an AI provider in AI settings]", model=None):
    prov, _, m = (model or "").partition("|")
    if prov not in PROV: prov, m = next(iter(PROV), ""), ""
    if not prov: return demo
    P = PROV[prov]
    r = P["client"].chat.completions.create(model=m or P["models"][0], temperature=0.2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content

@app.get("/api/skills")
def skills(): return list(SKILLS)

@app.get("/api/models")
def models():
    return [{"id": f"{p}|{m}", "label": f"{p} · {m}"} for p, d in PROV.items() for m in d["models"]] or [{"id": "demo", "label": "Demo mode (add an AI in AI settings)"}]

@app.post("/api/config")
def config(b: dict):
    if PUBLIC: return {"ok": False, "error": "Disabled on the public site."}
    name = re.sub(r"[^\w -]", "", b.get("name", ""))[:30].strip() or "custom"
    models = [m.strip() for m in b.get("models", "").split(",") if m.strip()]
    if not models: return {"ok": False, "error": "Add at least one model name"}
    old = PROV.get(name)
    reuse = old and not b.get("key") and (not b.get("base_url") or str(old["client"].base_url).rstrip("/") == b["base_url"].rstrip("/"))
    prev = old["models"] if reuse else None
    if reuse: old["models"] = models   # blank key on a saved provider: keep its stored key, change models only
    else: add_provider(name, b.get("base_url"), b.get("key"), models)
    if b.get("test", True):
        try:
            PROV[name]["client"].chat.completions.create(model=models[0], max_tokens=5, messages=[{"role": "user", "content": "ping"}])
        except Exception as e:
            if reuse: old["models"] = prev
            else: del PROV[name]
            return {"ok": False, "error": str(e)[:200]}
    return {"ok": True}

@app.post("/api/provider/remove")
def remove_provider(b: dict):
    if PUBLIC: return {"ok": False, "error": "Disabled on the public site."}
    PROV.pop(b.get("name"), None)
    return {"ok": True}

@app.post("/api/improve")
def improve(b: dict):
    miss = ", ".join(b.get("missing", []))
    out = llm("Rewrite the lawyer's drafting request to be specific and complete. Keep [placeholders] for unknown facts and never invent facts. Return only the rewritten request.",
              f"Request: {b['instruction']}\nMissing details to cover: {miss}", f"{b['instruction']} Also specify: {miss}.", b.get("model") or None)
    return {"instruction": out}

SKIP = ("whisper", "tts", "embed", "dall-e", "moderation", "guard", "audio", "transcribe", "realtime", "image", "babbage", "davinci", "rerank")

@app.post("/api/provider/models")
def provider_models(b: dict):
    if PUBLIC: return {"ok": False, "error": "Disabled on the public site."}
    # ask the provider which models this key can use
    try:
        if not b.get("key") and b.get("name") in PROV: c = PROV[b["name"]]["client"]
        else: c = OpenAI(api_key=b.get("key") or "none", base_url=b.get("base_url") or None, timeout=15)
        ids = sorted({m.id.removeprefix("models/") for m in c.models.list()})
    except Exception as e:
        return {"error": "Could not fetch the list (" + str(e)[:120] + "). Type the model names instead."}
    return {"models": [i for i in ids if not any(x in i.lower() for x in SKIP)]}

@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    S = sess(request)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as t:
        t.write(await file.read())
    md = to_markdown(t.name)
    os.unlink(t.name)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", md) if len(c.strip()) > 40]
    S.update(chunks=chunks, raw=len(md), bm25=BM25Okapi([tok(c) for c in chunks]) if chunks else None)
    S["style"] = llm("Analyse this legal document's drafting style in 6 bullets: tone, structure, numbering, clause patterns, terminology, formatting.", md[:4000], "Formal tone; numbered paragraphs (demo).")
    return {"chunks": len(chunks), "style": S["style"]}

class Gen(BaseModel):
    skill: str
    instruction: str
    model: str = ""

@app.post("/api/generate")
def generate(g: Gen, request: Request):
    S = sess(request)
    ctx = ""
    if S["bm25"]:                       # token management: only top-4 relevant chunks
        sc = S["bm25"].get_scores(tok(g.instruction))
        top = sorted(sorted(range(len(sc)), key=lambda i: -sc[i])[:4])
        ctx = "\n\n".join(S["chunks"][i] for i in top)
    system = SKILLS.get(g.skill, "") + RULES
    if S["style"]: system += "\nMatch this STYLE PROFILE:\n" + S["style"]
    prompt = (f"Reference excerpts:\n{ctx}\n\n" if ctx else "") + "Task: " + g.instruction
    return {"draft": llm(system, prompt, DEMO, g.model or None), "tokens": {"full": S["raw"] // 4, "sent": len(ctx) // 4}}

@app.post("/api/edit")
def edit(b: dict):
    out = llm("You are a legal editor. Apply the instruction and return the full revised document only." + RULES,
              f"Instruction: {b['instruction']}\n\nDocument:\n{b['draft']}",
              b["draft"] + f"\n\n[demo edit: {b['instruction']}]", b.get("model") or None)
    return {"draft": out}

# ---------- File reading (works with or without markitdown) ----------
def to_markdown(path):
    ext = Path(path).suffix.lower()
    try:
        from markitdown import MarkItDown
        return MarkItDown().convert(path).text_content
    except Exception:
        pass
    if ext == ".docx": return "\n\n".join(p.text for p in Document(path).paragraphs)
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(pg.extract_text() or "" for pg in PdfReader(path).pages)
    return Path(path).read_text(encoding="utf-8", errors="ignore")

# ---------- Citation verification ----------
CASE = re.compile(r"AIR\s*\d{4}\s*SC\s*\d+|\(\d{4}\)\s*\d+\s*SCC\s*\d+")
ACT = re.compile(r"[Ss]ection\s+\d+[A-Za-z]?\s+of\s+the\s+([A-Z][A-Za-z ]+?Act)")
norm = lambda s: re.sub(r"\s+", "", s).upper()

def kanoon(c):  # best-effort live check
    try:
        r = requests.get("https://indiankanoon.org/search/", params={"formInput": f'"{c}"'},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        return r.ok and "result_title" in r.text
    except Exception:
        return False

def item(c, st, note):
    return {"cite": c, "status": st, "note": note, "link": "https://indiankanoon.org/search/?formInput=" + quote(c)}

def name_span(text, pos):
    # index where the case name written just before a citation starts (pos if none found)
    before = text[max(0, pos - 90):pos]
    m = list(re.finditer(r"\s[vV]s?\.?\s", before))
    if not m: return pos
    left = before[:m[-1].start()].split(" ")
    i = len(left)
    while i > 0 and (left[i-1].lstrip("*_")[:1].isupper() or left[i-1] in ("of", "the")) and len(left) - i < 4: i -= 1
    return pos - len(before) + len(" ".join(left[:i])) + (1 if i else 0)

@app.post("/api/verify")
def verify(b: dict):
    text, out, seen = b["text"], [], set()
    for m in CASE.finditer(text):
        c = m.group(0)
        if norm(c) in seen: continue
        seen.add(norm(c))
        hit = next((a for a in AUTH["cases"] if norm(a["cite"]) == norm(c)), None)
        before = text[max(0, m.start() - 90):m.start()].lower()
        if hit and hit["key"] in before: r = item(c, "verified", "Matches " + hit["name"])
        elif hit:
            r = item(c, "partial", f"This citation belongs to {hit['name']}, but the draft names a different case")
            r["fix"] = {"old": text[name_span(text, m.start()):m.end()], "new": hit["name"] + ", " + c}
        elif kanoon(c): r = item(c, "partial", "Possible match on Indian Kanoon, confirm manually")
        else: r = item(c, "unverified", "Not in local database or Indian Kanoon: possible hallucination")
        out.append(r)
    for m in ACT.finditer(text):
        sec, act = re.search(r"\d+[A-Za-z]?", m.group(0)).group(0), m.group(1).strip()
        known = AUTH["sections"].get(act)
        if known and sec in known: out.append(item(m.group(0), "verified", "Act and section found in database"))
        elif known: out.append(item(m.group(0), "partial", "Act exists; section not in database, check manually"))
        else: out.append(item(m.group(0), "unverified", "Act not recognised"))
    return {"results": out}

@app.get("/api/snippet")
def snippet(cite: str):
    # best-effort: first Indian Kanoon search hit for this citation
    try:
        h = requests.get("https://indiankanoon.org/search/", params={"formInput": f'"{cite}"'},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=8).text
        t = re.search(r'class="result_title"[^>]*>\s*<a[^>]*>(.*?)</a>', h, re.S)
        s = re.search(r'class="headline"[^>]*>(.*?)</div>', h, re.S)
        strip = lambda x: re.sub(r"<[^>]+>|\s+", " ", x).strip()
        if t: return {"title": strip(t.group(1)), "snippet": strip(s.group(1)) if s else ""}
    except Exception:
        pass
    return {"title": "", "snippet": ""}

# ---------- Similar cases: supporting and adverse authorities ----------
def load_cases():
    return json.loads((BASE / "cases.json").read_text(encoding="utf-8"))   # re-read each time, so edits need no restart

STOP = set("section sections court matter grant granted judicial case cases draft application applications petition order orders applicant respondent state india union high under shall would should could which where there their about being have been this that with from into upon also such other than those these your".split())

@app.post("/api/similar")
def similar(b: dict):
    cases = load_cases()
    q = [t for t in tok(b["text"]) if len(t) > 3 and t not in STOP]
    docs = [tok(" ".join([c["name"], c["topic"], c["holding"], " ".join(c.get("issues", []))])) for c in cases]
    sc = BM25Okapi(docs).get_scores(q)
    out = {"supporting": [], "adverse": []}
    for i in sorted(range(len(cases)), key=lambda i: -sc[i]):
        c = cases[i]
        if c.get("stance") not in out or sc[i] <= 0 or len(out[c["stance"]]) >= 4: continue
        why = ", ".join(sorted(set(q) & set(docs[i]), key=len, reverse=True)[:4])
        out[c["stance"]].append({**c, "why": why, "link": "https://indiankanoon.org/search/?formInput=" + quote(c["cite"])})
    return out

@app.post("/api/distinguish")
def distinguish(b: dict):
    c = next((x for x in load_cases() if x["name"] == b["case"]), None)
    if not c: return {"text": ""}
    out = llm("You are an Indian advocate. Write ONE short paragraph, in the style of written submissions, respectfully distinguishing the given case from our matter. Use ONLY the holding provided, cite no other case, and write [UNVERIFIED] where unsure. Output the paragraph only.",
              f"Our draft:\n{b['draft'][:3000]}\n\nCase: {c['name']}, {c['cite']}\nHolding: {c['holding']}",
              f"The decision in {c['name']}, {c['cite']}, turned on its own facts, and the holding there ({c['holding']}) does not govern the present matter because [state the distinguishing facts here].",
              b.get("model") or None)
    return {"text": out}

# ---------- Public launch safety ----------
PUBLIC = os.getenv("PUBLIC_MODE", "") == "1"   # set PUBLIC_MODE=1 on a public server
HITS = defaultdict(deque)
LIMITED = ("/api/generate", "/api/edit", "/api/improve", "/api/distinguish", "/api/correct", "/api/translate", "/api/upload", "/api/command", "/api/transcribe")
SESS = {}   # one private reference-document store per visitor (cookie "sid")

def sess(request):
    sid = request.cookies.get("sid", "")
    if not sid: return {"chunks": [], "bm25": None, "raw": 0, "style": ""}
    if sid not in SESS:
        if len(SESS) > 200: SESS.pop(next(iter(SESS)))
        SESS[sid] = {"chunks": [], "bm25": None, "raw": 0, "style": ""}
    return SESS[sid]

@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path in LIMITED:
        ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
        q, now = HITS[ip], time.time()
        while q and now - q[0] > 3600: q.popleft()
        if len(q) >= int(os.getenv("RATE_PER_HOUR", "30")):
            return JSONResponse({"detail": "Rate limit reached. Please try again in a while."}, status_code=429)
        q.append(now)
    resp = await call_next(request)
    if "sid" not in request.cookies:
        resp.set_cookie("sid", uuid.uuid4().hex, httponly=True, samesite="lax", max_age=86400 * 30)
    return resp

@app.get("/api/public")
def public_flag(): return {"public": PUBLIC}

# ---------- Document tools: Correction Review + Translate ----------
_ocr = None

def read_any(path):
    # PDF, DOCX, TXT (via to_markdown), PPTX (python-pptx) and images (OCR)
    global _ocr
    ext = Path(path).suffix.lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            from rapidocr_onnxruntime import RapidOCR
            _ocr = _ocr or RapidOCR()
            res, _ = _ocr(path)
            return "\n".join(r[1] for r in (res or []))
        if ext == ".pptx":
            from pptx import Presentation
            out = []
            for i, sl in enumerate(Presentation(path).slides, 1):
                out.append(f"## Slide {i}")
                for sh in sl.shapes:
                    if sh.has_text_frame: out.append(sh.text_frame.text)
                    if getattr(sh, "has_table", False) and sh.has_table:
                        out += [" | ".join(c.text for c in row.cells) for row in sh.table.rows]
                if sl.has_notes_slide: out.append("Notes: " + sl.notes_slide.notes_text_frame.text)
            return "\n\n".join(out)
    except ImportError:
        if PUBLIC: raise RuntimeError("This file type is not enabled on the public site.")
        raise RuntimeError("Reading " + ext + " files needs a package. Run: python -m pip install " + ("python-pptx" if ext == ".pptx" else "rapidocr-onnxruntime"))
    return to_markdown(path)

async def get_text(file, text):
    if file is not None and file.filename:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as t:
            data = await file.read()
            if len(data) > 10_000_000: raise RuntimeError("File too large (10 MB maximum).")
            t.write(data)
        try: return read_any(t.name)
        finally: os.unlink(t.name)
    return text

SYS_REVIEW = "You are a senior Indian advocate reviewing a legal document for completeness and correctness. Compare it with what such a document normally needs under Indian law and practice. Return ONLY JSON of this shape: {\"doc_type\": str, \"summary\": str (2 sentences), \"present\": [{\"item\": str, \"evidence\": str (short quote or location)}], \"missing\": [{\"item\": str, \"why\": str, \"suggestion\": str}], \"issues\": [{\"text\": str (short quote), \"problem\": str, \"fix\": str}]}. Rules: list as present only what you can see in the text and quote the evidence; never invent case citations or section numbers, write 'verify' when unsure; at most 12 items per list."
DEMO_REVIEW = '{"doc_type":"Sample (demo mode)","summary":"Connect an AI in AI settings to review your own document.","present":[{"item":"Parties named","evidence":"Page 1"}],"missing":[{"item":"Governing law and jurisdiction clause","why":"Fixes which law and courts apply.","suggestion":"Add a clause naming Indian law and the competent courts."}],"issues":[{"text":"sample text","problem":"Sample issue","fix":"Sample fix"}]}'

@app.post("/api/correct")
async def correct(file: UploadFile = File(None), text: str = Form(""), option: str = Form(""), model: str = Form("")):
    try: src = await get_text(file, text)
    except Exception as e: return {"error": str(e)[:300]}
    if not src.strip(): return {"error": "No readable text found. For a scanned PDF, upload its pages as images."}
    hint = "unknown, detect it" if option.startswith("Auto") else option
    try:
        raw = llm(SYS_REVIEW + " Document type hint: " + hint + ".", src[:24000], DEMO_REVIEW, model or None)
        r = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        return {"error": "The AI did not return a usable review. Try again or pick a stronger model."}
    for k in ("present", "missing", "issues"): r[k] = [x for x in r.get(k, []) if isinstance(x, dict)]
    r["note"] = "Demo mode: sample output." if not PROV else "AI review can miss things or be wrong. A lawyer must check it before reliance." + (" Only the first 24,000 characters were reviewed." if len(src) > 24000 else "")
    g = lambda d, k: d.get(k, "")
    L = [f"# Correction review: {r.get('doc_type', 'Document')}", "", g(r, "summary"), "", "## What is there"]
    L += [f"- **{g(p, 'item')}**: {g(p, 'evidence')}" for p in r["present"]]
    L += ["", "## What is missing"] + [f"- **{g(m, 'item')}**: {g(m, 'why')} Suggestion: {g(m, 'suggestion')}" for m in r["missing"]]
    L += ["", "## Corrections to consider"] + [f"- **{g(x, 'problem')}** (\"{g(x, 'text')}\"): {g(x, 'fix')}" for x in r["issues"]]
    r["markdown"] = "\n".join(L)
    return r

@app.post("/api/translate")
async def translate(file: UploadFile = File(None), text: str = Form(""), option: str = Form("Hindi"), model: str = Form("")):
    try: src = await get_text(file, text)
    except Exception as e: return {"error": str(e)[:300]}
    if not src.strip(): return {"error": "No readable text found. For a scanned PDF, upload its pages as images."}
    paras = src.split("\n\n") if "\n\n" in src else src.split("\n")
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > 2500: chunks.append(cur); cur = ""
        cur += p + "\n\n"
    if cur.strip(): chunks.append(cur)
    sysm = f"You are a professional legal translator. Translate the text into {option}. Keep the Markdown structure (# headings, **bold**, lists, tables). Keep names, numbers, dates, section numbers and case citations exactly as written. Use the standard legal term in {option} where one exists. Output only the translation."
    try:
        out = [llm(sysm, c, "[demo mode: connect an AI in AI settings to translate] " + c[:200], model or None).strip() for c in chunks[:40]]
    except Exception as e:
        return {"error": "Translation failed: " + str(e)[:200]}
    return {"markdown": "\n\n".join(out) + ("\n\n[Document truncated: only the first 40 sections were translated.]" if len(chunks) > 40 else "")}

# ---------- Voice ----------
@app.get("/api/voice")
def voice_status():
    try:
        import faster_whisper  # noqa: F401
        return {"whisper": True}
    except Exception as e:
        return {"whisper": False, "error": str(e)[:150]}

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    global _whisper
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return {"error": "faster-whisper is not installed in this Python environment. Run: python -m pip install faster-whisper"}
    try:
        _whisper = _whisper or WhisperModel("base", compute_type="int8")   # downloads once (~150 MB)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as t:
            t.write(await audio.read())
        try:
            segs, _ = _whisper.transcribe(t.name)
            text = " ".join(x.text for x in segs).strip()
        finally:
            os.unlink(t.name)
        return {"text": text}
    except Exception as e:
        return {"error": "Whisper failed: " + str(e)[:200]}

@app.post("/api/command")
def command(b: dict):
    t = b["text"]
    sysmsg = ('Convert the lawyer\'s spoken command into JSON only: {"action":"generate|edit|verify|delete|export|skill","skill":"","instruction":""}. Skills: '
              + ", ".join(SKILLS))
    try:
        return json.loads(re.search(r"\{.*\}", llm(sysmsg, t, ""), re.S).group(0))
    except Exception:   # keyword fallback (also used in demo mode)
        l = t.lower()
        for k, a in [("delete", "delete"), ("clear", "delete"), ("verify", "verify"), ("check", "verify"),
                     ("export", "export"), ("download", "export"), ("generate", "generate"), ("draft", "generate")]:
            if k in l: return {"action": a, "instruction": t, "skill": ""}
        return {"action": "edit", "instruction": t, "skill": ""}

@app.post("/api/export")
def export(b: dict):
    d = Document()
    for line in b["draft"].split("\n"):
        if line.startswith("#"): d.add_heading(line.lstrip("# ").strip(), level=min(line.count("#"), 3))
        else:
            p = d.add_paragraph()
            for i, part in enumerate(line.split("**")): p.add_run(part).bold = (i % 2 == 1)
    buf = io.BytesIO(); d.save(buf); buf.seek(0)
    return StreamingResponse(buf, headers={"Content-Disposition": "attachment; filename=draft.docx"},
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

app.mount("/", StaticFiles(directory=BASE / "static", html=True))   # serves the frontend
