import os
import time
from typing import List, Optional, Dict, Set, Iterable

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


# ----------------------------
# Helpers
# ----------------------------

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Simple character-based chunker for X posts.
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

def get_all_x_posts_paginated(batch_size: int = 1000) -> List[Dict]:
    """
    Fetch ALL X posts from Supabase with non-empty full_text.

    This is NOT limited to 1000 total — it paginates until exhausted.
    Orders newest-first using scraped_at DESC.
    """
    all_rows: List[Dict] = []
    offset = 0

    while True:
        start = offset
        end = offset + batch_size - 1

        res = (
            supabase.table("x_posts")
            .select("id, full_text, scraped_at")
            .neq("full_text", "")
            .order("scraped_at", desc=True)
            .range(start, end)
            .execute()
        )

        batch = res.data or []
        if not batch:
            break

        all_rows.extend(batch)
        print(f"Fetched {len(batch)} x_posts (total so far: {len(all_rows)})")

        # If we got fewer than batch_size rows, we're done
        if len(batch) < batch_size:
            break

        offset += batch_size

    return all_rows


def get_existing_embedded_x_post_ids_paginated(batch_size: int = 1000) -> Set[str]:
    """
    Fetch ALL post_ids that already have embeddings in x_openai_post_chunks.
    Accumulate into a set to avoid per-post existence queries.
    """
    existing: Set[str] = set()
    offset = 0

    while True:
        start = offset
        end = offset + batch_size - 1

        res = (
            supabase.table("x_openai_post_chunks")
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
            sleep_s = 2 ** (attempt - 1)
            print(f"  Embedding error (attempt {attempt}/5): {e}. Sleeping {sleep_s}s...")
            time.sleep(sleep_s)

    raise last_err  # type: ignore


# ----------------------------
# Supabase Insert (batch)
# ----------------------------

def insert_x_openai_chunks_batch(rows: List[Dict], batch_size: int = 200):
    """
    Insert chunk rows into x_openai_post_chunks in batches.
    """
    if not rows:
        return

    for batch in batched(rows, batch_size):
        supabase.table("x_openai_post_chunks").insert(batch).execute()


# ----------------------------
# Main
# ----------------------------

def main():
    print("Fetching all X posts...")
    posts = get_all_x_posts_paginated(batch_size=1000)
    print(f"\nTotal X posts fetched with non-empty text: {len(posts)}")

    print("\nFetching existing embedded post_ids...")
    existing_post_ids = get_existing_embedded_x_post_ids_paginated(batch_size=1000)
    print(f"\nTotal unique X post_ids already embedded: {len(existing_post_ids)}")

    posts_to_embed = []
    for post in posts:
        post_id = post["id"]
        full_text = (post.get("full_text") or "").strip()

        if not full_text:
            continue

        if post_id in existing_post_ids:
            continue

        posts_to_embed.append(post)

    print(f"\nX posts missing embeddings: {len(posts_to_embed)}")

    embedded_posts_count = 0
    total_chunks_inserted = 0

    for idx, post in enumerate(posts_to_embed):
        post_id = post["id"]
        full_text = (post.get("full_text") or "").strip()

        chunks = chunk_text(full_text, max_chars=1000)
        if not chunks:
            continue

        print(f"[{idx+1}/{len(posts_to_embed)}] x_post_id={post_id} -> embedding {len(chunks)} chunks")

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

        insert_x_openai_chunks_batch(rows_to_insert, batch_size=200)

        embedded_posts_count += 1
        total_chunks_inserted += len(rows_to_insert)

    print("\nDone embedding X OpenAI chunks.")
    print(f"  Newly embedded X posts (OpenAI): {embedded_posts_count}")
    print(f"  Total chunks inserted: {total_chunks_inserted}")


if __name__ == "__main__":
    main()