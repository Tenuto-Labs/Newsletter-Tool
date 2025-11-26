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

# ---------- CORS (so the browser frontend can call this API) ---------- #

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5500")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ---------- #


class AskRequest(BaseModel):
    question: str
    k: int = 8  # how many chunks to retrieve


class AskResponse(BaseModel):
    answer: str
    used_chunks: List[str]


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
    """
    lines = []
    for i, ch in enumerate(chunks):
        meta = f"(chunk {i+1}, posted_at={ch.get('posted_at')}, url={ch.get('linkedin_url')})"
        text = (ch["chunk_text"] or "").strip().replace("\n", " ")
        lines.append(f"{meta}\n{text}\n")
    return "\n\n".join(lines)


# ---------- FastAPI endpoint ---------- #


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1) Retrieve relevant chunks
    chunks = retrieve_chunks(req.question, req.k)

    if not chunks:
        return AskResponse(
            answer="I couldn't find any relevant posts in the database.",
            used_chunks=[],
        )

    context = build_context(chunks)

    # 2) System-style instructions
    system_prompt = (
        "You are an assistant that answers questions ONLY based on the provided "
        "excerpts from Ethan Mollick's LinkedIn posts (and other authors in the DB, if present).\n"
        "- If something is not supported by the excerpts, say you don't know.\n"
        "- When possible, mention chunk numbers or LinkedIn URLs you used."
    )

    # 3) Single user prompt string
    user_prompt = (
        f"{system_prompt}\n\n"
        f"User question:\n{req.question}\n\n"
        f"Here are relevant excerpts from the posts:\n\n"
        f"{context}\n\n"
        "Using ONLY the above excerpts, answer the user's question clearly and concisely."
    )

    # 4) Call Gemini
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
    )

    response = model.generate_content(user_prompt)
    answer_text = response.text

    return AskResponse(
        answer=answer_text,
        used_chunks=[c["chunk_text"] for c in chunks],
    )