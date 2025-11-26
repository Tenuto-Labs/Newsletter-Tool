import os
import textwrap
from typing import List

import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # Gemini API key

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL = "text-embedding-004"  # or another embedding model


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Simple character-based chunker for Ethan's posts.

    For more sophistication you could use sentence-level splitting
    and respect paragraph boundaries.
    """
    text = text.strip()
    if not text:
        return []

    # naive fixed-size chunks
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def get_all_posts():
    """
    Fetch all posts from Supabase.
    Adjust 'range' if you have more than 1000 posts.
    """
    res = (
        supabase.table("posts")
        .select("id, full_text")
        .neq("full_text", "")  # ignore empty
        .execute()
    )
    return res.data


def embed_text(text: str) -> List[float]:
    """
    Call Gemini embeddings API for a single text chunk.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",  # hint that this is a doc chunk
    )
    # result.embeddings[0].values for some versions, or result["embedding"] for others
    # We'll handle the common case:
    emb = result["embedding"] if isinstance(result, dict) else result.embeddings[0].values
    return emb


def insert_chunk(post_id: str, chunk_index: int, chunk_text: str, embedding: List[float]):
    """
    Insert a single chunk row into post_chunks.
    Supabase's PostgREST can accept an array for vector if you cast properly on the DB side,
    but the simplest way is to define an RPC or use raw SQL.

    For now, we'll send as text and cast to vector in SQL via a stored procedure.
    However, Supabase's Python client does support inserting if you pass a list
    and the PostgREST / pgvector integration is enabled.

    We'll assume Supabase can accept list -> vector.
    """
    supabase.table("post_chunks").insert(
        {
            "post_id": post_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "embedding": embedding,
            "embedding_model": EMBEDDING_MODEL,
        }
    ).execute()


def main():
    posts = get_all_posts()
    print(f"Embedding {len(posts)} posts...")

    for idx, post in enumerate(posts):
        post_id = post["id"]
        full_text = post["full_text"] or ""
        full_text = full_text.strip()
        if not full_text:
            continue

        chunks = chunk_text(full_text)
        print(f"[{idx+1}/{len(posts)}] post_id={post_id} -> {len(chunks)} chunks")

        for c_idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            insert_chunk(post_id, c_idx, chunk, emb)

    print("Done embedding all posts.")


if __name__ == "__main__":
    main()
