import os
from typing import List

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# Embedding model (already used in your pipeline)
EMBEDDING_MODEL = "text-embedding-004"

# Chat model from your list_models() output
CHAT_MODEL = "models/gemini-2.5-pro"

app = FastAPI(title="Ethan Mollick RAG Assistant")

# ---------- CORS (allow everything for prototype) ---------- #

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow any origin (GitHub Pages, localhost, etc.)
    allow_credentials=False,  # we don't use cookies/sessions
    allow_methods=["*"],      # allow all HTTP methods
    allow_headers=["*"],      # allow all request headers
)

# ---------- Health check/root ---------- #


@app.get("/")
def root():
    return {"status": "ok", "service": "ethan-assistant"}


# ---------- Pydantic models ---------- #


class AskRequest(BaseModel):
    question: str
    # slightly higher default k so the model has more to work with
    k: int = 10


class AskResponse(BaseModel):
    answer: str


# ---------- Helpers ---------- #


def embed_query(query: str) -> List[float]:
    """
    Embed the user's question using the embedding model.
    """
    result = genai.embed_content(
      model=EMBEDDING_MODEL,
      content=query,
      task_type="retrieval_query",
    )
    # handle both dict-style and object-style responses
    emb = result["embedding"] if isinstance(result, dict) else result.embeddings[0].values
    return emb


def retrieve_chunks(query: str, k: int) -> List[dict]:
    """
    Call match_post_chunks via Supabase RPC to get relevant chunks.
    """
    q_emb = embed_query(query)
    rpc_response = supabase.rpc(
        "match_post_chunks",
        {
            "query_embedding": q_emb,
            "match_count": k,
        },
    ).execute()
    return rpc_response.data or []


def build_context(chunks: List[dict]) -> str:
    """
    Turn retrieved chunks into a context block for Gemini.
    We keep light metadata (like posted_at) purely for the model's reasoning.
    """
    lines = []
    for i, ch in enumerate(chunks):
        posted_at = ch.get("posted_at") or "unknown date"
        url = ch.get("linkedin_url") or "unknown url"
        text = (ch.get("chunk_text") or "").strip()
        if not text:
            continue

        # This is internal structure for the model, not something we tell it to repeat verbatim.
        lines.append(
            f"Excerpt {i+1} (posted_at={posted_at}, url={url}):\n{text}\n"
        )
    return "\n\n".join(lines)


# ---------- FastAPI endpoint ---------- #


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1) Retrieve relevant chunks
    chunks = retrieve_chunks(req.question, req.k)

    if not chunks:
        return AskResponse(
            answer=(
                "I couldn't find any relevant posts in the database for that question. "
                "Try rephrasing or asking about a topic Ethan has written about, "
                "like AI, work, management, or education."
            )
        )

    context = build_context(chunks)

    # 2) System-style instructions
    system_prompt = (
        "You are an assistant that answers questions using ONLY the provided excerpts from "
        "Ethan Mollick's LinkedIn posts (and other authors in the DB, if present).\n\n"
        "Your job is to:\n"
        "- Synthesize and connect ideas across the excerpts.\n"
        "- Write rich, well-structured explanations (several paragraphs or a mix of paragraphs and bullet points).\n"
        "- Highlight key themes, tensions, and implications for practice when relevant.\n"
        "- Stay faithful to Ethan's views and tone as reflected in the excerpts.\n"
        "- You may generalize and extrapolate as long as it is consistent with the excerpts, "
        "but do NOT invent views that clearly contradict them.\n\n"
        "Very important:\n"
        "- Do NOT mention 'chunks', 'excerpts', IDs, or say things like 'in excerpt 3' or 'the first chunk'.\n"
        "- Do NOT talk about the retrieval process, embeddings, or the database.\n"
        "- Just answer as if you are summarizing and interpreting Ethan's public writing."
    )

    # 3) Single user prompt string
    user_prompt = (
        f"{system_prompt}\n\n"
        f"User question:\n{req.question}\n\n"
        f"Here are relevant excerpts from the posts:\n\n"
        f"{context}\n\n"
        "Using ONLY the above excerpts as your factual grounding, write a detailed, thoughtful answer "
        "to the user's question. Aim for depth and nuance rather than brevity. Where it helps, you can "
        "organize the answer into clear sections or bullet points."
    )

    # 4) Call Gemini
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
    )

    response = model.generate_content(user_prompt)
    answer_text = response.text

    return AskResponse(
        answer=answer_text,
    )