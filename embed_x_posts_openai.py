# embed_x_posts_openai.py
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Must match: public.x_openai_post_chunks.embedding vector(1536)
EMBEDDING_MODEL = "text-embedding-3-small"


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Simple character-based chunker for X posts.
    """
    text = (text or "").strip()
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def get_all_x_posts(limit: int = 1000):
    """
    Fetch all X posts from Supabase with non-empty full_text.
    Uses pagination to avoid default API row limits.
    """
    all_rows = []
    offset = 0

    while True:
        res = (
            supabase.table("x_posts")
            .select("id, full_text")
            .neq("full_text", "")
            .range(offset, offset + limit - 1)
            .execute()
        )
        batch = res.data or []
        all_rows.extend(batch)

        if len(batch) < limit:
            break
        offset += limit

    return all_rows


def x_post_has_openai_chunks(post_id: str) -> bool:
    """
    Check whether this X post already has any chunks in x_openai_post_chunks.
    """
    res = (
        supabase.table("x_openai_post_chunks")
        .select("id")
        .eq("post_id", post_id)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def embed_text(text: str) -> List[float]:
    """
    Call OpenAI embeddings API for a single text chunk.
    Includes a small retry loop for transient failures/rate limits.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, 6):
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            last_err = e
            # basic backoff: 1s, 2s, 4s, 8s, 16s
            sleep_s = 2 ** (attempt - 1)
            print(f"  Embedding error (attempt {attempt}/5): {e}. Sleeping {sleep_s}s...")
            time.sleep(sleep_s)

    raise last_err  # type: ignore


def insert_x_openai_chunk(post_id: str, chunk_index: int, chunk_text: str, embedding: List[float]):
    """
    Insert a single chunk row into x_openai_post_chunks.
    """
    supabase.table("x_openai_post_chunks").insert(
        {
            "post_id": post_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "embedding": embedding,
            "embedding_model": EMBEDDING_MODEL,
        }
    ).execute()


def main():
    posts = get_all_x_posts()
    print(f"Found {len(posts)} X posts with non-empty text.")

    embedded_count = 0
    skipped_count = 0

    for idx, post in enumerate(posts):
        post_id = post["id"]
        full_text = (post.get("full_text") or "").strip()
        if not full_text:
            continue

        if x_post_has_openai_chunks(post_id):
            print(f"[{idx+1}/{len(posts)}] x_post_id={post_id} -> already has OpenAI chunks, skipping.")
            skipped_count += 1
            continue

        chunks = chunk_text(full_text)
        if not chunks:
            continue

        print(f"[{idx+1}/{len(posts)}] x_post_id={post_id} -> embedding {len(chunks)} OpenAI chunks")

        for c_idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            insert_x_openai_chunk(post_id, c_idx, chunk, emb)

        embedded_count += 1

    print("Done embedding X OpenAI chunks.")
    print(f"  Newly embedded X posts (OpenAI): {embedded_count}")
    print(f"  Skipped (already had OpenAI chunks): {skipped_count}")


if __name__ == "__main__":
    main()