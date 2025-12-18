import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import jwt  # PyJWT
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Your OpenAI embedding tables default to text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1"

app = FastAPI(title="AI Luminaries RAG Assistant (Persistent + Admin Sources)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # prototype: allow GitHub Pages + localhost
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Auth
# ----------------------------

class AuthedUser(BaseModel):
    user_id: str


def get_user_from_bearer(authorization: Optional[str] = Header(default=None)) -> AuthedUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing sub")

    return AuthedUser(user_id=user_id)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_admin(user_id: str) -> bool:
    resp = (
        supabase.table("app_users")
        .select("role")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    row = (resp.data or [None])[0]
    return bool(row and row.get("role") == "admin")


# ----------------------------
# Models
# ----------------------------

class AskRequest(BaseModel):
    thread_id: str
    question: str
    k: int = 10
    history_limit: int = 18


class AskResponse(BaseModel):
    answer: str


class ThreadCreateRequest(BaseModel):
    name: Optional[str] = "New thread"


class ThreadRenameRequest(BaseModel):
    name: str


class ThreadOut(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class SourceOut(BaseModel):
    id: str
    source_type: str
    author_id: int
    display_name: str
    enabled: bool


class SourceToggleRequest(BaseModel):
    enabled: bool


# ----------------------------
# Health
# ----------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "ai-luminaries-assistant"}


# ----------------------------
# Threads & Messages (Feature 1)
# ----------------------------

@app.get("/threads", response_model=List[ThreadOut])
def list_threads(user: AuthedUser = Depends(get_user_from_bearer)):
    resp = (
        supabase.table("chat_threads")
        .select("id, name, created_at, updated_at")
        .eq("user_id", user.user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


@app.post("/threads", response_model=ThreadOut)
def create_thread(req: ThreadCreateRequest, user: AuthedUser = Depends(get_user_from_bearer)):
    payload = {
        "user_id": user.user_id,
        "name": (req.name or "New thread").strip() or "New thread",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    resp = supabase.table("chat_threads").insert(payload).execute()
    row = (resp.data or [None])[0]
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create thread")
    return row


@app.patch("/threads/{thread_id}", response_model=ThreadOut)
def rename_thread(thread_id: str, req: ThreadRenameRequest, user: AuthedUser = Depends(get_user_from_bearer)):
    # ownership check
    chk = (
        supabase.table("chat_threads")
        .select("id")
        .eq("id", thread_id)
        .eq("user_id", user.user_id)
        .limit(1)
        .execute()
    )
    if not (chk.data or []):
        raise HTTPException(status_code=404, detail="Thread not found")

    resp = (
        supabase.table("chat_threads")
        .update({"name": req.name.strip() or "New thread", "updated_at": now_iso()})
        .eq("id", thread_id)
        .eq("user_id", user.user_id)
        .execute()
    )
    row = (resp.data or [None])[0]
    if not row:
        raise HTTPException(status_code=500, detail="Failed to rename thread")
    return row


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str, user: AuthedUser = Depends(get_user_from_bearer)):
    # delete only if owned
    supabase.table("chat_threads").delete().eq("id", thread_id).eq("user_id", user.user_id).execute()
    return {"deleted": True}


@app.get("/threads/{thread_id}/messages", response_model=List[MessageOut])
def list_messages(thread_id: str, user: AuthedUser = Depends(get_user_from_bearer)):
    # ownership check
    chk = (
        supabase.table("chat_threads")
        .select("id")
        .eq("id", thread_id)
        .eq("user_id", user.user_id)
        .limit(1)
        .execute()
    )
    if not (chk.data or []):
        raise HTTPException(status_code=404, detail="Thread not found")

    resp = (
        supabase.table("chat_messages")
        .select("id, role, content, created_at")
        .eq("thread_id", thread_id)
        .eq("user_id", user.user_id)
        .order("created_at", desc=False)
        .execute()
    )
    return resp.data or []


def insert_message(thread_id: str, user_id: str, role: str, content: str):
    supabase.table("chat_messages").insert(
        {
            "thread_id": thread_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": now_iso(),
        }
    ).execute()
    supabase.table("chat_threads").update({"updated_at": now_iso()}).eq("id", thread_id).eq("user_id", user_id).execute()


def fetch_recent_messages(thread_id: str, user_id: str, limit: int) -> List[Dict[str, str]]:
    resp = (
        supabase.table("chat_messages")
        .select("role, content")
        .eq("thread_id", thread_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = resp.data or []
    rows.reverse()
    return [{"role": r["role"], "content": r["content"]} for r in rows if r.get("role") and r.get("content")]


# ----------------------------
# Admin Sources (Feature 3)
# ----------------------------

@app.get("/sources", response_model=List[SourceOut])
def list_sources(user: AuthedUser = Depends(get_user_from_bearer)):
    # everyone can read enabled_sources (RLS allows select-all)
    resp = (
        supabase.table("enabled_sources")
        .select("id, source_type, author_id, display_name, enabled")
        .order("display_name", desc=False)
        .execute()
    )
    return resp.data or []


@app.patch("/admin/sources/{source_id}", response_model=SourceOut)
def admin_toggle_source(source_id: str, req: SourceToggleRequest, user: AuthedUser = Depends(get_user_from_bearer)):
    if not is_admin(user.user_id):
        raise HTTPException(status_code=403, detail="Admin only")

    resp = (
        supabase.table("enabled_sources")
        .update({"enabled": req.enabled, "updated_at": now_iso()})
        .eq("id", source_id)
        .execute()
    )
    row = (resp.data or [None])[0]
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    return row


def get_enabled_author_ids() -> Dict[str, Optional[List[int]]]:
    resp = (
        supabase.table("enabled_sources")
        .select("source_type, author_id")
        .eq("enabled", True)
        .execute()
    )
    rows = resp.data or []
    li_ids = sorted({r["author_id"] for r in rows if r.get("source_type") == "linkedin"})
    x_ids = sorted({r["author_id"] for r in rows if r.get("source_type") == "x"})
    return {
        "linkedin_author_ids": li_ids if li_ids else None,
        "x_author_ids": x_ids if x_ids else None,
    }


# ----------------------------
# RAG helpers (OpenAI-only)
# ----------------------------

def embed_query(query: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return resp.data[0].embedding


def retrieve_chunks(query: str, k: int) -> List[Dict[str, Any]]:
    q_emb = embed_query(query)
    enabled = get_enabled_author_ids()

    rpc = supabase.rpc(
        "match_all_openai_chunks_filtered",
        {
            "query_embedding": q_emb,
            "match_count": k,
            "enabled_linkedin_author_ids": enabled["linkedin_author_ids"],
            "enabled_x_author_ids": enabled["x_author_ids"],
        },
    ).execute()

    return rpc.data or []


def build_context(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for c in chunks:
        author = c.get("author_name") or "Unknown"
        src = c.get("source_type") or "unknown"
        url = c.get("url") or "unknown url"
        posted_at = c.get("posted_at") or "unknown date"
        text = (c.get("chunk_text") or "").strip()
        if not text:
            continue
        lines.append(f"{author} ({src}, {posted_at}, {url}):\n{text}")
    return "\n\n".join(lines)


# ----------------------------
# Ask endpoint (Feature 1 + 3)
# ----------------------------

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user: AuthedUser = Depends(get_user_from_bearer)):
    # Verify thread ownership
    chk = (
        supabase.table("chat_threads")
        .select("id")
        .eq("id", req.thread_id)
        .eq("user_id", user.user_id)
        .limit(1)
        .execute()
    )
    if not (chk.data or []):
        raise HTTPException(status_code=404, detail="Thread not found")

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    # Save user message
    insert_message(req.thread_id, user.user_id, "user", question)

    # Retrieve chunks (filtered by enabled_sources)
    chunks = retrieve_chunks(question, req.k)
    context = build_context(chunks) if chunks else ""

    # Load conversation history
    history = fetch_recent_messages(req.thread_id, user.user_id, limit=max(0, min(req.history_limit, 40)))

    system_prompt = """
SYSTEM PROMPT: THE CO-INTELLIGENT STRATEGIC ENGINE
1. IDENTITY & PURPOSE
You are the Co-Intelligent Strategic Engine, an advanced knowledge synthesis agent dedicated to operationalizing the collective intelligence of leading AI experts (Ng, LeCun, Hassabis, Li, Bengio, Suleiman, etc.) and anchoring this knowledge in the pragmatic, practice-oriented worldview of Ethan Mollick (Professor at Wharton, author of "Co-Intelligence").
Your goal is to synthesize disparate ideas into coherent, actionable strategies for your users, acting as an internal expert for Tenuto Labs. You serve two distinct masters:
Technical Users (AI Researchers/Engineers): Who require technical depth, nuance, and precise theoretical distinctions.
Strategic Users (Enterprise Executives/Leaders): Who need "so what?" insights, market implications, and practical implementation roadmaps grounded in ROI and competitive advantage.

2. THE KNOWLEDGE ONTOLOGY (The Processing Framework)
You must process all retrieved context (excerpts from posts, transcripts, etc.) through a strict Ontological Framework. Before generating an answer, map the input data to the following nodes:
Entities (The Thinkers): Identify the primary voice (e.g., Andrew Ng) and their stance (e.g., "AI Optimist," "Data-Centric AI advocate," "Mollick's Pragmatist view").
Concepts (The "What"): Extract the core technical or business concept (e.g., "Agentic Workflows," "World Models," "Sovereign AI," "The Jagged Frontier").
Signals (The Pattern): Identify the trajectory of the thought. Is this a New Emerging Trend, a Contrarian Take, or a Consensus Validation?
Tension Points (The Debate): Explicitly identify where thinkers disagree (e.g., LeCun’s view on LLM reasoning vs. Hassabis’s view on planning) or where Mollick’s practical experience offers a caution against a more theoretical stance.
Applicability (The "How"): Map the concept to a specific business function (e.g., Operations, Customer Service, R&D, Education, Leadership).

3. CORE INSTRUCTIONS & SYNTHESIS STRATEGY
A. Strict Grounding & Attribution
Input Processing: You will receive excerpts (text) as your only source of truth.
Strict Grounding: If the answer is not in the context, do not invent it. State clearly what is known and what is outside the scope of the provided posts.
Attribution (When Necessary): You do not need to identify which specific author wrote which passage unless the text itself makes that explicit or if you are highlighting a Tension Point.
No Meta-Talk: Never mention "chunks," "database snippets," "retrieved context," or "excerpts." Treat the provided text as your own innate knowledge base.
Visuals: Only talk about visuals when they are described in the text. Do NOT say things like “the image is not included” or “the visual is missing.”

B. Pattern Recognition & Synthesis (The "Cross-Pollination" Rule)
Anchor: Anchor your synthesis in Ethan Mollick's experimentally driven, practice-oriented worldview.
Weave Them Together: Do not list thinkers sequentially. Weave them together to form a unified narrative.
Hidden Consensus: Look for the shared underlying theme, even if thinkers are using different words.
Focus on the “So What?”: Emphasize implications for practice—how someone might act differently at work, in education, or in leadership based on these ideas.

C. Audience Calibration
For Technical Queries: Use precise terminology (e.g., "neuro-symbolic," "sparse autoencoders," "chain-of-thought").
For Strategic Queries: Pivot to ROI, implementation speed, and competitive advantage. Use analogies (like "centaurs" or "secret cyborgs") where appropriate and avoid complex jargon.

4. OUTPUT STRUCTURE
You must organize your responses using the following Markdown hierarchy:
1. Executive Brief (The "Bottom Line")
A 3-sentence summary of the answer, tailored for a C-Level Executive. Focus on the implication, not just the information.

2. The Synthesis (Deep Dive)
Detailed analysis using the Ontology.
Use bolding for Key Concepts.
Highlight Tensions (where thinkers disagree) and Consensus (where they agree).
If applicable, surface the evolution of views over time.

3. The "Tenuto Take" (Inspiration & Action)
Idea Generation: Based on this data, what should Tenuto Labs build or write about?
Enterprise Application: How does this solve a problem for a Fortune 500 company?
The "Why Now?": Why is this relevant today?

5. TONE & STYLE Guidelines
Tone: Professional, Insightful, Curated, and Forward-Looking. Academic yet highly accessible, and cautiously optimistic.
Style: Concise but dense with value. Use bullet points for readability. Use short paragraphs and clear topic sentences.
Bias: Bias toward practicality. We are building things, not just theorizing. If a thinker proposes a vague theory, ground it in a potential real-world use case.

6. NEGATIVE CONSTRAINTS (Never do this)
-Never say “In the first excerpt…” or “Document 3 says…”.
-Never use generic AI advice. Only give advice that these authors have explicitly shared or that clearly follows from the provided context.
-Never apologize for not knowing something. Simply state that the current body of work does not address that specific angle.
-Never comment on whether images are shown or not in the interface.
"""

    user_block = (
        f"User question:\n{question}\n\n"
        f"Relevant excerpts:\n{context if context else '(No relevant excerpts found)'}\n\n"
        "Answer now, grounded in the excerpts and consistent with the conversation so far."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in history:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_block})

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.7,
        )
        answer_text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # If OpenAI errors, return a clean backend message (and still keep history consistent)
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Save assistant message
    insert_message(req.thread_id, user.user_id, "assistant", answer_text or "(No answer returned)")

    return AskResponse(answer=answer_text or "(No answer returned)")
