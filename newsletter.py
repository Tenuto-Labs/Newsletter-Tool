import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr

load_dotenv()

# ---- Supabase + Gemini config ---- #

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

LOOKBACK_DAYS_DEFAULT = 7        # "recent" window for newsletter themes
SCRAPE_WINDOW_DAYS = 60          # how far back to look in scraped_at to build a candidate set
MAX_POSTS_DEFAULT = 80
CHAT_MODEL = "models/gemini-2.5-pro"

genai.configure(api_key=GOOGLE_API_KEY)

# ---- Email config ---- #

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")  # required if sending email
SMTP_PASS = os.environ.get("SMTP_PASS")  # required if sending email
NEWSLETTER_FROM_NAME = os.environ.get("NEWSLETTER_FROM_NAME", "AI Newsletter")
NEWSLETTER_RECIPIENTS_ENV = os.environ.get("NEWSLETTER_RECIPIENTS", "")


def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ---------- Date helpers: estimate real posted_at from scraped_at + relative text ---------- #

def parse_relative_offset(raw: str) -> timedelta | None:
    """
    Parse a relative time expression like:
        "3w • 3 weeks ago • Visible to anyone..."
        "2d • 2 days ago • ..."
        "4mo • 4 months ago • ..."
        "1y • 1 year ago • ..."
    and return a timedelta approximation.

    We keep it intentionally simple and conservative.
    """
    if not raw:
        return None

    raw_lower = raw.lower()

    # First try compact form like "3w", "2d", "4mo", "1y"
    m = re.search(r"(\d+)\s*(d|w|mo|y)\b", raw_lower)
    if m:
        value = int(m.group(1))
        unit = m.group(2)
    else:
        # Try long form like "3 days", "2 weeks", "4 months", "1 year"
        m2 = re.search(
            r"(\d+)\s*(day|days|week|weeks|month|months|year|years)\b",
            raw_lower,
        )
        if not m2:
            return None
        value = int(m2.group(1))
        unit_word = m2.group(2)
        if "day" in unit_word:
            unit = "d"
        elif "week" in unit_word:
            unit = "w"
        elif "month" in unit_word:
            unit = "mo"
        else:
            unit = "y"

    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    if unit == "mo":
        # approximate month as 30 days
        return timedelta(days=30 * value)
    if unit == "y":
        # approximate year as 365 days
        return timedelta(days=365 * value)

    return None


def estimate_posted_datetime(scraped_at_str: str | None, posted_raw: str | None) -> datetime:
    """
    Use scraped_at + relative posted_at string to estimate an approximate posted datetime.

    If we can't parse anything, we fall back to scraped_at itself.
    """
    # Parse scraped_at from Supabase (ISO 8601 string) to datetime
    now = datetime.now(timezone.utc)
    if scraped_at_str:
        try:
            # handle possible "Z" suffix
            s = scraped_at_str.replace("Z", "+00:00")
            scraped_dt = datetime.fromisoformat(s)
        except Exception:
            scraped_dt = now
    else:
        scraped_dt = now

    if not posted_raw:
        return scraped_dt

    offset = parse_relative_offset(posted_raw)
    if not offset:
        return scraped_dt

    # If posted_at_raw says "3w ago" at scrape time, then approximate:
    # posted_dt ≈ scraped_dt - 3 weeks
    return scraped_dt - offset


# ---------- Fetch & filter posts ---------- #

def fetch_recent_posts(
    supabase: Client,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    max_posts: int = MAX_POSTS_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Fetch posts whose *estimated posted time* is within the last `lookback_days` days.

    Strategy:
      1. Grab posts whose scraped_at is within a broader window (SCRAPE_WINDOW_DAYS).
      2. For each, estimate a real-ish posted_at datetime using scraped_at + posted_at text.
      3. Filter in Python to only keep posts with estimated_posted_at >= now - lookback_days.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    # Step 1: broad superset by scraped_at
    scrape_start = now - timedelta(days=SCRAPE_WINDOW_DAYS)
    scrape_start_iso = scrape_start.isoformat()
    now_iso = now.isoformat()

    res = (
        supabase.table("posts")
        .select(
            "id, full_text, scraped_at, posted_at, linkedin_url, author_id, authors(display_name)"
        )
        .gte("scraped_at", scrape_start_iso)
        .lte("scraped_at", now_iso)
        .order("scraped_at", desc=True)
        .execute()
    )
    rows = res.data or []

    # Step 2–3: estimate posted_at and filter
    recent: List[Dict[str, Any]] = []

    for p in rows:
        scraped_at_str = p.get("scraped_at")
        posted_raw = p.get("posted_at") or ""
        est_posted_dt = estimate_posted_datetime(scraped_at_str, posted_raw)

        # Attach for later use in formatting if desired
        p["_estimated_posted_at"] = est_posted_dt.isoformat()

        if est_posted_dt >= cutoff:
            recent.append(p)
            if len(recent) >= max_posts:
                break

    return recent


def build_newsletter_source(posts: List[Dict[str, Any]]) -> str:
    """
    Build a compact text block grouped by author.

    Each post includes:
      - author name (if available)
      - estimated posted_at (for recency)
      - raw posted_at text
      - linkedin_url
      - truncated full_text
    """
    by_author: Dict[str, List[Dict[str, Any]]] = {}
    for p in posts:
        author_info = p.get("authors") or {}
        author_name = author_info.get("display_name") or "Unknown author"
        by_author.setdefault(author_name, []).append(p)

    lines: List[str] = []

    for author_name, items in by_author.items():
        # Sort by estimated posted_at desc (most recent first)
        items_sorted = sorted(
            items,
            key=lambda x: x.get("_estimated_posted_at") or "",
            reverse=True,
        )
        lines.append(f"=== AUTHOR: {author_name} ===")

        for p in items_sorted:
            est_posted = p.get("_estimated_posted_at") or "unknown_estimated_posted_at"
            posted_at_raw = p.get("posted_at") or "unknown posted_at text"
            url = p.get("linkedin_url") or "no url"
            text = (p.get("full_text") or "").strip()

            if len(text) > 1200:
                text = text[:1200] + " [...]"

            lines.append(
                f"- Post (estimated_posted_at={est_posted}, posted_at_raw={posted_at_raw}, url={url}):\n"
                f"{text}\n"
            )

        lines.append("")  # blank line between authors

    return "\n".join(lines)


# ---------- System prompt: PLAIN TEXT output, no Markdown ---------- #

NEWSLETTER_SYSTEM_PROMPT = """
You are an editorial assistant creating a concise weekly AI trends one-sheet.

Audience:
- Busy executives and practitioners who follow AI but do not track every individual post.
- They want synthesis of what is happening, not a raw feed of links.

Your sources:
- A set of recent posts (roughly within the last 1–2 weeks) from Ethan Mollick and other AI leaders
  such as Andrew Ng, Yann LeCun, Mustafa Suleyman, Demis Hassabis, Fei-Fei Li, and Yoshua Bengio.
- You will see them grouped by author with approximate dates and LinkedIn URLs.

Important formatting instructions:
- OUTPUT MUST BE PLAIN TEXT, not Markdown.
- Do NOT use '#' characters for headings.
- Do NOT use '*' characters for bullet points or bold/italics.
- Do NOT wrap text in any other markup syntax (no **bold**, _italics_, etc.).
- Instead, use simple uppercase or title-case headings and hyphen bullets. For example:

THIS WEEK IN AI: Short Title

Overview
One short paragraph summarizing the week.

Theme 1 – Name of Theme
- What is happening
- Who is saying what
- Why it matters

Theme 2 – ...
...

Notable Posts
- Author Name: one-line takeaway (link: https://...)

Core instructions:
- Identify 3–5 key themes or trends in AI reflected in these posts
  (e.g., multimodal capabilities, enterprise AI adoption, AI & education, governance, model evaluations, etc.).
- For each theme:
  - Explain what is happening in plain language.
  - Attribute ideas to authors when appropriate (e.g., “Mollick argues…”, “Ng emphasizes…”).
  - Highlight concrete implications for organizations, educators, or leaders.
- If multiple authors discuss the same theme, synthesize their perspectives.
- If there are notable differences in emphasis or viewpoint, briefly contrast them.

Constraints:
- Do not invent facts beyond what can reasonably be inferred from the source posts.
- Do not mention that you are summarizing “scraped data” or “database excerpts”—just write as a normal newsletter.
- Aim for about 600–900 words.
"""


def generate_newsletter_text(
    posts: List[Dict[str, Any]],
    model_name: str = CHAT_MODEL,
) -> str:
    """
    Given a list of recent posts, call Gemini to generate a newsletter-style
    plain-text document summarizing key themes.
    """
    if not posts:
        return (
            "AI WEEKLY\n\n"
            "No recent posts were found in the selected time window."
        )

    source_text = build_newsletter_source(posts)

    user_prompt = (
        f"{NEWSLETTER_SYSTEM_PROMPT}\n\n"
        f"Here are the recent posts:\n\n"
        f"{source_text}"
    )

    model = genai.GenerativeModel(model_name=model_name)
    resp = model.generate_content(user_prompt)
    text = resp.text or ""
    return text


def derive_subject_from_text(text: str) -> str:
    """
    Use the first non-empty line as the email subject, falling back to a default if needed.
    """
    if not text.strip():
        return "AI Weekly Newsletter"

    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]  # just in case it's very long

    return "AI Weekly Newsletter"


def parse_recipient_list(env_value: str, cli_value: str | None) -> List[str]:
    """
    Combine recipients from env and CLI, split on commas, and dedupe.
    """
    recipients: List[str] = []

    if env_value:
        recipients.extend([r.strip() for r in env_value.split(",") if r.strip()])

    if cli_value:
        recipients.extend([r.strip() for r in cli_value.split(",") if r.strip()])

    # dedupe
    seen = set()
    unique = []
    for r in recipients:
        if r not in seen:
            seen.add(r)
            unique.append(r)

    return unique


def send_newsletter_email(
    subject: str,
    body: str,
    recipients: List[str],
):
    """
    Send the newsletter as a plain-text email to the given recipients.
    """
    if not recipients:
        raise ValueError("No recipients provided for newsletter email.")

    if not SMTP_USER or not SMTP_PASS:
        raise RuntimeError(
            "SMTP_USER and SMTP_PASS must be set in the environment to send email."
        )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr((NEWSLETTER_FROM_NAME, SMTP_USER))
    msg["To"] = ", ".join(recipients)

    print(f"Sending newsletter email to: {', '.join(recipients)}")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, recipients, msg.as_string())

    print("Email sent successfully.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a weekly AI trends one-sheet from recent LinkedIn posts."
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS_DEFAULT,
        help="How many days back (true post age) to consider recent (default: 7).",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=MAX_POSTS_DEFAULT,
        help="Maximum number of posts to include after filtering (default: 80).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="newsletter.md",
        help="Output file path for the generated newsletter text.",
    )
    parser.add_argument(
        "--email-to",
        type=str,
        default="",
        help="Additional comma-separated recipient emails (on top of NEWSLETTER_RECIPIENTS env).",
    )

    args = parser.parse_args()

    supabase = get_supabase_client()
    posts = fetch_recent_posts(
        supabase, lookback_days=args.lookback_days, max_posts=args.max_posts
    )

    print(f"Fetched {len(posts)} recent posts (by estimated posted time).")

    text = generate_newsletter_text(posts)

    # Write to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nNewsletter written to {args.output}\n")
    print("---- Preview (first ~40 lines) ----")
    lines = text.splitlines()
    preview = "\n".join(lines[:40])
    print(preview)
    print("---- End preview ----")

    # Always attempt to send an email
    recipients = parse_recipient_list(NEWSLETTER_RECIPIENTS_ENV, args.email_to)
    if not recipients:
        print(
            "ERROR: No recipients found. "
            "Set NEWSLETTER_RECIPIENTS in your .env or pass --email-to on the command line."
        )
        return

    subject = derive_subject_from_text(text)
    send_newsletter_email(subject, text, recipients)


if __name__ == "__main__":
    main()