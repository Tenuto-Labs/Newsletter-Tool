import os
from typing import List

from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Match the vector dimension in openai_post_chunks (vector(1536))
EMBEDDING_MODEL = "text-embedding-3-small"


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Simple character-based chunker for posts.

    For more sophistication you could use sentence-level splitting
    and respect paragraph boundaries.
    """
    text = text.strip()
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def get_all_posts():
    """
    Fetch all posts from Supabase that have non-empty full_text.
    """
    res = (
        supabase.table("posts")
        .select("id, full_text")
        .neq("full_text", "")
        .execute()
    )
    return res.data


def post_has_openai_chunks(post_id: str) -> bool:
    """
    Check whether this post already has any chunks in openai_post_chunks.
    """
    res = (
        supabase.table("openai_post_chunks")
        .select("id")
        .eq("post_id", post_id)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def embed_text(text: str) -> List[float]:
    """
    Call OpenAI embeddings API for a single text chunk.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def insert_openai_chunk(post_id: str, chunk_index: int, chunk_text: str, embedding: List[float]):
    """
    Insert a single chunk row into openai_post_chunks.
    """
    supabase.table("openai_post_chunks").insert(
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

        if post_has_openai_chunks(post_id):
            print(f"[{idx+1}/{len(posts)}] post_id={post_id} -> already has OpenAI chunks, skipping.")
            skipped_count += 1
            continue

        chunks = chunk_text(full_text)
        print(f"[{idx+1}/{len(posts)}] post_id={post_id} -> embedding {len(chunks)} OpenAI chunks")

        for c_idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            insert_openai_chunk(post_id, c_idx, chunk, emb)

        embedded_count += 1

    print("Done embedding OpenAI chunks.")
    print(f"  Newly embedded posts (OpenAI): {embedded_count}")
    print(f"  Skipped (already had OpenAI chunks): {skipped_count}")


if __name__ == "__main__":
    main()