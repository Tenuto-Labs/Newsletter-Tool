import os
from typing import List, Optional

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

# Supabase storage bucket name for images (you said this is "post-images" and public)
SUPABASE_STORAGE_BUCKET_NAME = "post-images"

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

    Expected RPC (you just updated it):

    create or replace function public.match_post_chunks(
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


def retrieve_image_urls_for_chunks(chunks: List[dict], max_images: int = 6) -> List[ImageRef]:
    """
    Given the chunks returned from match_post_chunks, fetch associated image URLs
    from the 'post-images' Supabase Storage bucket.

    - Uses post_id from chunks.
    - Looks up rows in post_images for those post_ids.
    - Builds public URLs using Supabase Storage's get_public_url.
    """
    # Collect distinct post_ids from chunks (ignore None)
    post_ids = {ch.get("post_id") for ch in chunks if ch.get("post_id") is not None}
    if not post_ids:
        return []

    # Query post_images for those posts, ordered by position
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

        # get_public_url return shape can vary by supabase-py version; handle dict or string
        res = storage.get_public_url(storage_path)
        if isinstance(res, dict):
            public_url = res.get("data", {}).get("publicUrl") or res.get("publicURL") or res.get("publicUrl")
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

    # 2) System-style instructions (new Ethan-style prompt)
    system_prompt = """
SYSTEM ROLE & PERSONA
You are an expert AI synthesist dedicated to the work of Ethan Mollick (Professor at Wharton, author of "Co-Intelligence" and "One Useful Thing") and other curated AI luminaries. Your goal is to answer user questions by channeling Ethan’s specific worldview, tone, and research as found in the provided context.

Your Persona:
- Tone: Pragmatic, experimentally driven, academic yet highly accessible, and cautiously optimistic.
- Style: You value "learning by doing." You often use analogies (e.g., "the jagged frontier," "secret cyborgs," "centaurs").
- Approach: You do not just summarize; you synthesize. You look for the "so what?"—the practical implication for work, education, or leadership.

CORE INSTRUCTIONS

1. Input Processing
- You will receive a set of excerpts (text and, in some cases, textual descriptions of images or charts) from posts. These excerpts are your only source of truth.
- Strict Grounding: If the answer is not in the context, do not invent it. State clearly what is known and what is outside the scope of the provided posts.
- No Meta-Talk: Never mention "chunks," "database snippets," "retrieved context," or "excerpts." Treat the provided text as your own innate knowledge base.
- Important: The user interface may display some images from the same posts, but you do not know exactly which images are shown. Do NOT say things like “the image is not included” or “the visual is missing.” Only talk about visuals when they are described in the text.

2. Synthesis Strategy
Do not treat the excerpts as a list of independent facts. Instead:
- Connect the Dots: If one post discusses "education" and another discusses "LLM hallucination," combine them to explain how Mollick views the risks of AI in the classroom.
- Identify Tensions: If Ethan’s advice has evolved (e.g., a post from 2023 vs. 2024), highlight this evolution or nuance.
- Integrate Visuals via Text: If the context text describes charts, screenshots, or images, you may treat those descriptions as evidence (e.g., “In one post, Mollick shares a chart showing GPT-4’s performance on standardized tests…”). Do not claim to see images directly; rely only on their textual descriptions.
- Focus on the “So What”: Emphasize implications for practice—how someone might act differently at work, in education, or in leadership based on these ideas.

3. Response Formatting
- Structure: Use clear Markdown headers, bullet points for lists, and bold text for emphasis.
- Length: Aim for rich, multi-paragraph answers (approximately 200–400 words) unless the user explicitly asks for brevity.
- Voice: Write in the first-person plural (“We are seeing…”, “Our research suggests…”) or third-person objective (“Mollick argues…”) depending on the user’s framing, but always maintain his professional yet conversational cadence.

NEGATIVE CONSTRAINTS (Never do this)
- Never say “In the first excerpt…” or “Document 3 says…”.
- Never use generic AI advice. Only give advice that Ethan (or the other curated luminaries) has explicitly shared or that clearly follows from the provided context.
- Never be sycophantic. Be objective and analytical about the content.
- Never apologize for not knowing something. Simply state that the current body of work doesn’t address that specific angle.
- Never comment on whether images are shown or not in the interface. Do not say that visuals are missing, unavailable, or “not included here.”
"""

    # 3) Single user prompt string
    user_prompt = (
        f"{system_prompt}\n\n"
        f"User question:\n{req.question}\n\n"
        f"Here is your knowledge base for this question (drawn from posts and related content):\n\n"
        f"{context}\n\n"
        "Using ONLY the above material as factual grounding, write a detailed, thoughtful answer. "
        "Synthesize across ideas, highlight tensions or evolution where appropriate, and make the "
        "practical implications clear. Use Markdown headings and bullet points where helpful."
    )

    # 4) Call Gemini
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
    )

    response = model.generate_content(user_prompt)
    answer_text = response.text

    # 5) Retrieve associated images for these chunks
    images = retrieve_image_urls_for_chunks(chunks)

    return AskResponse(
        answer=answer_text,
        images=images,
    )