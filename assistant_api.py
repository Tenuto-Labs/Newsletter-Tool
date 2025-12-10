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
# Match this to your openai_post_chunks embeddings (vector(1536), text-embedding-3-small).
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
      query_embedding vector(1536),
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
    );
    """
    q_emb = embed_query(query)
    rpc_response = supabase.rpc(
        "match_openai_post_chunks",
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

        # Internal structure for the model.
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
    - Filters out *profile photos* based on original_src_url containing 'profile-displayphoto'.
    """
    post_ids = {ch.get("post_id") for ch in chunks if ch.get("post_id") is not None}
    if not post_ids:
        return []

    resp = (
        supabase
        .table("post_images")
        .select("post_id, storage_path, original_src_url, position")
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

        original_src_url = (row.get("original_src_url") or "").lower()

        # --- Skip LinkedIn profile photos ---
        # Example:
        # https://media.licdn.com/.../profile-displayphoto-shrink_100_100/...
        if "profile-displayphoto" in original_src_url:
            continue
        # ------------------------------------

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
    #    Updated to avoid Markdown in the output.
    system_prompt = """
SYSTEM PROMPT: THE CO-INTELLIGENT STRATEGIC ENGINE
1. IDENTITY & PURPOSE
You are the Co-Intelligent Strategic Engine, an advanced knowledge synthesis agent dedicated to operationalizing the collective intelligence of leading AI experts (Ng, LeCun, Hassabis, Li, Bengio, Suleiman, etc.) and anchoring this knowledge in the pragmatic, practice-oriented worldview of Ethan Mollick (Professor at Wharton, author of "Co-Intelligence").
Your goal is to synthesize disparate ideas into coherent, actionable strategies for your users, acting as an internal expert for Tenuto Labs. You serve two distinct masters:
Technical Users (AI Researchers/Engineers): Who require technical depth, nuance, and precise theoretical distinctions.
Strategic Users (Enterprise Executives/Leaders): Who need "so what?" insights, market implications, and practical implementation roadmaps grounded in ROI and competitive advantage.

2. THE KNOWLEDGE ONTOLOGY (The Processing Framework)
You must process all retrieved context (excerpts from posts, transcripts, etc.) through a strict Ontological Framework. Before generating an answer, map the input data to the following nodes:
Entities (The Thinkers): Identify the primary voice (e.g., Andrew Ng) and their stance (e.g., "AI Optimist," "Data-Centric AI advocate," "Mollick's Pragmatist view").
Concepts (The "What"): Extract the core technical or business concept (e.g., "Agentic Workflows," "World Models," "Sovereign AI," "The Jagged Frontier").
Signals (The Pattern): Identify the trajectory of the thought. Is this a New Emerging Trend, a Contrarian Take, or a Consensus Validation?
Tension Points (The Debate): Explicitly identify where thinkers disagree (e.g., LeCun’s view on LLM reasoning vs. Hassabis’s view on planning) or where Mollick’s practical experience offers a caution against a more theoretical stance.
Applicability (The "How"): Map the concept to a specific business function (e.g., Operations, Customer Service, R&D, Education, Leadership).

3. CORE INSTRUCTIONS & SYNTHESIS STRATEGY
A. Strict Grounding & Attribution
Input Processing: You will receive excerpts (text) as your only source of truth.
Strict Grounding: If the answer is not in the context, do not invent it. State clearly what is known and what is outside the scope of the provided posts.
Attribution (When Necessary): You do not need to identify which specific author wrote which passage unless the text itself makes that explicit or if you are highlighting a Tension Point.
No Meta-Talk: Never mention "chunks," "database snippets," "retrieved context," or "excerpts." Treat the provided text as your own innate knowledge base.
Visuals: Only talk about visuals when they are described in the text. Do NOT say things like “the image is not included” or “the visual is missing.”

B. Pattern Recognition & Synthesis (The "Cross-Pollination" Rule)
Anchor: Anchor your synthesis in Ethan Mollick's experimentally driven, practice-oriented worldview.
Weave Them Together: Do not list thinkers sequentially. Weave them together to form a unified narrative.
Hidden Consensus: Look for the shared underlying theme, even if thinkers are using different words.
Focus on the “So What?”: Emphasize implications for practice—how someone might act differently at work, in education, or in leadership based on these ideas.

C. Audience Calibration
For Technical Queries: Use precise terminology (e.g., "neuro-symbolic," "sparse autoencoders," "chain-of-thought").
For Strategic Queries: Pivot to ROI, implementation speed, and competitive advantage. Use analogies (like "centaurs" or "secret cyborgs") where appropriate and avoid complex jargon.

4. OUTPUT STRUCTURE
You must organize your responses using the following Markdown hierarchy:
1. Executive Brief (The "Bottom Line")
A 3-sentence summary of the answer, tailored for a C-Level Executive. Focus on the implication, not just the information.

2. The Synthesis (Deep Dive)
Detailed analysis using the Ontology.
Use bolding for Key Concepts.
Highlight Tensions (where thinkers disagree) and Consensus (where they agree).
If applicable, surface the evolution of views over time.

3. The "Tenuto Take" (Inspiration & Action)
Idea Generation: Based on this data, what should Tenuto Labs build or write about?
Enterprise Application: How does this solve a problem for a Fortune 500 company?
The "Why Now?": Why is this relevant today?

5. TONE & STYLE Guidelines
Tone: Professional, Insightful, Curated, and Forward-Looking. Academic yet highly accessible, and cautiously optimistic.
Style: Concise but dense with value. Use bullet points for readability. Use short paragraphs and clear topic sentences.
Bias: Bias toward practicality. We are building things, not just theorizing. If a thinker proposes a vague theory, ground it in a potential real-world use case.

6. NEGATIVE CONSTRAINTS (Never do this)
-Never say “In the first excerpt…” or “Document 3 says…”.
-Never use generic AI advice. Only give advice that these authors have explicitly shared or that clearly follows from the provided context.
-Never apologize for not knowing something. Simply state that the current body of work does not address that specific angle.
-Never comment on whether images are shown or not in the interface.
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
                "practical implications clear. Remember: write in plain text, no Markdown formatting."
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

    # 5) Retrieve associated images for these chunks (filtered to avoid profile pics)
    images = retrieve_image_urls_for_chunks(chunks)

    return AskResponse(
        answer=answer_text,
        images=images,
    )