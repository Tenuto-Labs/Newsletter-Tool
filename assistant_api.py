import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Supabase storage bucket name for images (public)
SUPABASE_STORAGE_BUCKET_NAME = "post-images"

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- OpenAI models ----
# Your openai_post_chunks table + embed script use text-embedding-3-small (1536-dim)
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1"

app = FastAPI(title="AI Luminaries RAG Assistant")

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
    return {"status": "ok", "service": "ai-luminaries-assistant"}


# ---------- Pydantic models ---------- #


class ImageRef(BaseModel):
    url: str
    alt: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    # slightly higher default k so the model has more to work with
    k: int = 10


class AskResponse(BaseModel):
    answer: str
    images: List[ImageRef] = []


# ---------- Helpers ---------- #


def embed_query(query: str) -> List[float]:
    """
    Embed the user's question using the OpenAI embedding model.

    NOTE: Your openai_post_chunks table should also use this same model
    (text-embedding-3-small) for good similarity behavior.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return resp.data[0].embedding


def retrieve_chunks(query: str, k: int) -> List[dict]:
    """
    Call match_openai_post_chunks via Supabase RPC to get relevant chunks.

    Expected RPC signature:

    create or replace function public.match_openai_post_chunks(
      query_embedding vector,
      match_count int
    )
    returns table (
      id uuid,
      post_id uuid,
      chunk_index int,
      chunk_text text,
      similarity double precision,
      posted_at text,
      linkedin_url text
    )
    """
    q_emb = embed_query(query)
    rpc_response = supabase.rpc(
        "match_openai_post_chunks",    # OpenAI-specific RPC
        {
            "query_embedding": q_emb,
            "match_count": k,
        },
    ).execute()
    return rpc_response.data or []


def build_context(chunks: List[dict]) -> str:
    """
    Turn retrieved chunks into a context block for the LLM.
    We keep light metadata (like posted_at) purely for the model's reasoning.
    """
    lines = []
    for i, ch in enumerate(chunks):
        posted_at = ch.get("posted_at") or "unknown date"
        url = ch.get("linkedin_url") or "unknown url"
        text = (ch.get("chunk_text") or "").strip()
        if not text:
            continue

        lines.append(
            f"Excerpt {i+1} (posted_at={posted_at}, url={url}):\n{text}\n"
        )
    return "\n\n".join(lines)


def retrieve_image_urls_for_chunks(chunks: List[dict], max_images: int = 6) -> List[ImageRef]:
    """
    Given the chunks returned from match_openai_post_chunks, fetch associated image URLs
    from the 'post-images' Supabase Storage bucket.

    - Uses post_id from chunks.
    - Looks up rows in post_images for those post_ids.
    - Builds public URLs using Supabase Storage's get_public_url.
    """
    post_ids = {ch.get("post_id") for ch in chunks if ch.get("post_id") is not None}
    if not post_ids:
        return []

    resp = (
        supabase
        .table("post_images")
        .select("post_id, storage_path, position")
        .in_("post_id", list(post_ids))
        .order("position", desc=False)
        .execute()
    )

    rows = resp.data or []
    image_refs: List[ImageRef] = []

    storage = supabase.storage.from_(SUPABASE_STORAGE_BUCKET_NAME)

    for row in rows:
        storage_path = row.get("storage_path")
        if not storage_path:
            continue

        res = storage.get_public_url(storage_path)
        if isinstance(res, dict):
            public_url = (
                res.get("data", {}).get("publicUrl")
                or res.get("publicURL")
                or res.get("publicUrl")
            )
        else:
            public_url = str(res)

        if not public_url:
            continue

        image_refs.append(ImageRef(url=public_url, alt="Image from source post"))

        if len(image_refs) >= max_images:
            break

    return image_refs


# ---------- FastAPI endpoint ---------- #


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1) Retrieve relevant chunks
    chunks = retrieve_chunks(req.question, req.k)

    if not chunks:
        return AskResponse(
            answer=(
                "The current body of posts doesn't directly address that question. "
                "Try asking about topics these authors are known to explore—such as AI, work, "
                "management, education, experimentation, or emerging AI capabilities."
            ),
            images=[],
        )

    context = build_context(chunks)

    # 2) System-style instructions (multi-author Ethan-centered prompt)
    system_prompt = """
SYSTEM ROLE & PERSONA
You are an expert AI synthesist dedicated to the work of Ethan Mollick (Professor at Wharton, author of "Co-Intelligence" and "One Useful Thing") and a curated set of AI luminaries, including Andrew Ng, Yann LeCun, Mustafa Suleyman, Demis Hassabis, Fei-Fei Li, and Yoshua Bengio.

Your goal is to answer user questions by drawing on this shared body of posts and writing. You anchor your synthesis in Ethan Mollick’s experimentally driven, practice-oriented worldview, while also incorporating perspectives from the other authors where relevant.

Your Persona:
- Tone: Pragmatic, experimentally driven, academic yet highly accessible, and cautiously optimistic.
- Style: You value "learning by doing." You often use analogies (e.g., "the jagged frontier," "secret cyborgs," "centaurs").
- Approach: You do not just summarize; you synthesize. You look for the "so what?"—the practical implication for work, education, or leadership.

CORE INSTRUCTIONS

1. Input Processing
- You will receive a set of excerpts (text and, in some cases, textual descriptions of images or charts) from posts. These excerpts are your only source of truth.
- Treat the excerpts as coming from Ethan Mollick and the other named AI luminaries; you do not need to identify which specific author wrote which passage unless the text itself makes that explicit.
- Strict Grounding: If the answer is not in the context, do not invent it. State clearly what is known and what is outside the scope of the provided posts.
- No Meta-Talk: Never mention "chunks," "database snippets," "retrieved context," or "excerpts." Treat the provided text as your own innate knowledge base.
- Important: The user interface may display some images from the same posts, but you do not know exactly which images are shown. Do NOT say things like “the image is not included” or “the visual is missing.” Only talk about visuals when they are described in the text.

2. Synthesis Strategy
Do not treat the excerpts as a list of independent facts. Instead:
- Connect the Dots: Combine ideas across authors. For example, if one passage discusses education and another discusses LLM hallucination or model capabilities, use them together to explain how these thinkers view AI risks and opportunities.
- Compare and Contrast: When the excerpts suggest differences in emphasis or perspective between authors (e.g., more cautious vs. more optimistic), briefly surface those tensions.
- Temporal Nuance: If advice or views have evolved over time, highlight that evolution.
- Integrate Visuals via Text: If the context text describes charts, screenshots, or images, you may treat those descriptions as evidence (e.g., “In one post, a chart shows GPT-4’s performance on standardized tests…”). Do not claim to see images directly; rely only on their textual descriptions.
- Focus on the “So What”: Emphasize implications for practice—how someone might act differently at work, in education, or in leadership based on these ideas.

3. Response Formatting
- Structure: Use clear Markdown headers, bullet points for lists, and bold text for emphasis.
- Length: Aim for rich, multi-paragraph answers (approximately 200–400 words) unless the user explicitly asks for brevity.
- Voice: Write in the first-person plural (“We are seeing…”, “Our research suggests…”) or third-person objective (“Mollick argues…”, “Ng emphasizes…”) depending on the user’s framing, while keeping a professional yet conversational cadence.

NEGATIVE CONSTRAINTS (Never do this)
- Never say “In the first excerpt…” or “Document 3 says…”.
- Never use generic AI advice. Only give advice that these authors have explicitly shared or that clearly follows from the provided context.
- Never be sycophantic. Be objective and analytical about the content.
- Never apologize for not knowing something. Simply state that the current body of work doesn’t address that specific angle.
- Never comment on whether images are shown or not in the interface. Do not say that visuals are missing, unavailable, or “not included here.”
"""

    # 3) Messages for OpenAI chat API
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User question:\n{req.question}\n\n"
                f"Here is your knowledge base for this question (drawn from posts and related content):\n\n"
                f"{context}\n\n"
                "Using ONLY the above material as factual grounding, write a detailed, thoughtful answer. "
                "Synthesize across ideas and authors, highlight tensions or evolution where appropriate, and make the "
                "practical implications clear. Use Markdown headings and bullet points where helpful."
            ),
        },
    ]

    # 4) Call OpenAI Chat Completions
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
    )
    answer_text = resp.choices[0].message.content

    # 5) Retrieve associated images for these chunks
    images = retrieve_image_urls_for_chunks(chunks)

    return AskResponse(
        answer=answer_text,
        images=images,
    )