import os
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
    Simple character-based chunker for posts.

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
    Fetch all posts from Supabase that have non-empty full_text.

    We will then decide per-post whether it needs embedding
    based on whether post_chunks already exist for that post_id.
    """
    res = (
        supabase.table("posts")
        .select("id, full_text")
        .neq("full_text", "")  # ignore empty
        .execute()
    )
    return res.data


def post_has_chunks(post_id: str) -> bool:
    """
    Check whether this post already has any chunks in post_chunks.

    This makes the script idempotent: if chunks exist, we skip re-embedding.
    """
    res = (
        supabase.table("post_chunks")
        .select("id")
        .eq("post_id", post_id)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def embed_text(text: str) -> List[float]:
    """
    Call Gemini embeddings API for a single text chunk.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",  # hint that this is a doc chunk
    )
    emb = result["embedding"] if isinstance(result, dict) else result.embeddings[0].values
    return emb


def insert_chunk(post_id: str, chunk_index: int, chunk_text: str, embedding: List[float]):
    """
    Insert a single chunk row into post_chunks.

    We assume Supabase + pgvector can accept a Python list for the vector column.
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
    print(f"Found {len(posts)} posts with non-empty text.")

    embedded_count = 0
    skipped_count = 0

    for idx, post in enumerate(posts):
        post_id = post["id"]
        full_text = (post["full_text"] or "").strip()
        if not full_text:
            continue

        # Skip posts that already have chunks
        if post_has_chunks(post_id):
            print(f"[{idx+1}/{len(posts)}] post_id={post_id} -> already has chunks, skipping.")
            skipped_count += 1
            continue

        chunks = chunk_text(full_text)
        print(f"[{idx+1}/{len(posts)}] post_id={post_id} -> embedding {len(chunks)} chunks")

        for c_idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            insert_chunk(post_id, c_idx, chunk, emb)

        embedded_count += 1

    print(f"Done embedding.")
    print(f"  Newly embedded posts: {embedded_count}")
    print(f"  Skipped (already had chunks): {skipped_count}")


if __name__ == "__main__":
    main()