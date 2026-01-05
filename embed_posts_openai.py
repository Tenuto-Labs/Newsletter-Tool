import os
import time
from typing import List, Dict, Set, Iterable, Optional

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


# ----------------------------
# Helpers
# ----------------------------

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Simple character-based chunker for posts.
    """
    text = (text or "").strip()
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def batched(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    """
    Yield successive batches from a list.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ----------------------------
# Supabase Fetching (Paginated)
# ----------------------------

def fetch_posts_paginated(batch_size: int = 1000) -> List[Dict]:
    """
    Fetch ALL posts with non-empty full_text using pagination.
    Orders newest-first using scraped_at DESC (since created_at doesn't exist).
    """
    all_posts: List[Dict] = []
    offset = 0

    while True:
        start = offset
        end = offset + batch_size - 1

        res = (
            supabase.table("posts")
            .select("id, full_text, scraped_at")
            .neq("full_text", "")
            .order("scraped_at", desc=True)
            .range(start, end)
            .execute()
        )

        batch = res.data or []
        if not batch:
            break

        all_posts.extend(batch)
        print(f"Fetched {len(batch)} posts (total so far: {len(all_posts)})")

        if len(batch) < batch_size:
            break

        offset += batch_size

    return all_posts


def fetch_existing_embedded_post_ids_paginated(batch_size: int = 1000) -> Set[str]:
    """
    Fetch ALL post_ids that already have chunks in openai_post_chunks.
    """
    existing: Set[str] = set()
    offset = 0

    while True:
        start = offset
        end = offset + batch_size - 1

        res = (
            supabase.table("openai_post_chunks")
            .select("post_id")
            .range(start, end)
            .execute()
        )

        batch = res.data or []
        if not batch:
            break

        for row in batch:
            pid = row.get("post_id")
            if pid:
                existing.add(pid)

        print(
            f"Fetched {len(batch)} chunk rows "
            f"(unique embedded post_ids so far: {len(existing)})"
        )

        if len(batch) < batch_size:
            break

        offset += batch_size

    return existing


# ----------------------------
# OpenAI Embeddings (with retry)
# ----------------------------

def embed_text(text: str, max_retries: int = 5) -> List[float]:
    """
    Call OpenAI embeddings API for a single text chunk.
    Includes retry/backoff for transient failures or rate limits.
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return resp.data[0].embedding

        except Exception as e:
            last_err = e
            sleep_s = 2 ** (attempt - 1)
            print(f"  Embedding error (attempt {attempt}/{max_retries}): {e}. Sleeping {sleep_s}s...")
            time.sleep(sleep_s)

    raise last_err  # type: ignore


# ----------------------------
# Supabase Insert (batch)
# ----------------------------

def insert_openai_chunks_batch(rows: List[Dict], batch_size: int = 200):
    """
    Insert chunk rows into openai_post_chunks in batches.
    """
    if not rows:
        return

    for batch in batched(rows, batch_size):
        supabase.table("openai_post_chunks").insert(batch).execute()


# ----------------------------
# Main
# ----------------------------

def main():
    print("Fetching posts...")
    posts = fetch_posts_paginated(batch_size=1000)
    print(f"\nTotal posts fetched with non-empty text: {len(posts)}")

    print("\nFetching existing embedded post_ids...")
    existing_post_ids = fetch_existing_embedded_post_ids_paginated(batch_size=1000)
    print(f"\nTotal unique post_ids already embedded: {len(existing_post_ids)}")

    # Only embed posts that don't yet have chunks
    posts_to_embed = []
    for post in posts:
        post_id = post["id"]
        full_text = (post.get("full_text") or "").strip()

        if not full_text:
            continue
        if post_id in existing_post_ids:
            continue

        posts_to_embed.append(post)

    print(f"\nPosts missing embeddings: {len(posts_to_embed)}")

    embedded_posts_count = 0
    total_chunks_inserted = 0

    for idx, post in enumerate(posts_to_embed):
        post_id = post["id"]
        full_text = (post.get("full_text") or "").strip()

        chunks = chunk_text(full_text, max_chars=1000)
        if not chunks:
            continue

        print(f"[{idx+1}/{len(posts_to_embed)}] post_id={post_id} -> embedding {len(chunks)} chunks")

        rows_to_insert: List[Dict] = []
        for c_idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            rows_to_insert.append(
                {
                    "post_id": post_id,
                    "chunk_index": c_idx,
                    "chunk_text": chunk,
                    "embedding": emb,
                    "embedding_model": EMBEDDING_MODEL,
                }
            )

        insert_openai_chunks_batch(rows_to_insert, batch_size=200)

        embedded_posts_count += 1
        total_chunks_inserted += len(rows_to_insert)

    print("\nDone embedding OpenAI chunks.")
    print(f"  Newly embedded posts: {embedded_posts_count}")
    print(f"  Total chunks inserted: {total_chunks_inserted}")


if __name__ == "__main__":
    main()