import os
from typing import List
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL = "text-embedding-004"

app = FastAPI(title="Ethan Mollick Semantic Search API")


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    chunk_text: str
    similarity: float
    posted_at: str | None = None
    linkedin_url: str | None = None


def embed_query(query: str) -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    emb = result["embedding"] if isinstance(result, dict) else result.embeddings[0].values
    return emb


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest):
    # 1) Embed the query
    q_emb = embed_query(req.query)

    # 2) Call Postgres function via Supabase RPC
    # Supabase will serialize q_emb (list[float]) to JSON; on the Postgres side,
    # the function expects extensions.vector(768). If this fails, you may need a small
    # wrapper function that casts from double precision[].
    rpc_response = supabase.rpc(
        "match_post_chunks",
        {
            "query_embedding": q_emb,
            "match_count": req.k,
        },
    ).execute()

    rows = rpc_response.data or []

    results: List[SearchResult] = []
    for row in rows:
        results.append(
            SearchResult(
                chunk_text=row["chunk_text"],
                similarity=row["similarity"],
                posted_at=row.get("posted_at"),
                linkedin_url=row.get("linkedin_url"),
            )
        )

    return results