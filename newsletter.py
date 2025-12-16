import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple

from supabase import create_client, Client
from dotenv import load_dotenv

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr

from openai import OpenAI

load_dotenv()

# ---- Supabase + OpenAI config ---- #

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

LOOKBACK_DAYS_DEFAULT = 7
SCRAPE_WINDOW_DAYS = 60
MAX_POSTS_DEFAULT = 150

# Choose any OpenAI chat model you want here (NOT tied to embedding model)
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Email config ---- #

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
NEWSLETTER_FROM_NAME = os.environ.get("NEWSLETTER_FROM_NAME", "AI Newsletter")
NEWSLETTER_RECIPIENTS_ENV = os.environ.get("NEWSLETTER_RECIPIENTS", "")


def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ---------- Date helpers (LinkedIn logic you already have) ---------- #

def parse_relative_offset(raw: str) -> timedelta | None:
    if not raw:
        return None

    raw_lower = raw.lower()

    m = re.search(r"(\d+)\s*(d|w|mo|y)\b", raw_lower)
    if m:
        value = int(m.group(1))
        unit = m.group(2)
    else:
        m2 = re.search(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)\b", raw_lower)
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
        return timedelta(days=30 * value)
    if unit == "y":
        return timedelta(days=365 * value)

    return None


def estimate_posted_datetime(scraped_at_str: str | None, posted_raw: str | None) -> datetime:
    now = datetime.now(timezone.utc)
    if scraped_at_str:
        try:
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

    return scraped_dt - offset


# ---------- Fetch posts (you can keep your existing logic; this is minimal) ---------- #

def fetch_recent_linkedin_posts(
    supabase: Client,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    max_posts: int = MAX_POSTS_DEFAULT,
) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    scrape_start = now - timedelta(days=SCRAPE_WINDOW_DAYS)
    res = (
        supabase.table("posts")
        .select("id, full_text, scraped_at, posted_at, linkedin_url, author_id, authors(display_name)")
        .gte("scraped_at", scrape_start.isoformat())
        .lte("scraped_at", now.isoformat())
        .order("scraped_at", desc=True)
        .execute()
    )
    rows = res.data or []

    recent: List[Dict[str, Any]] = []
    for p in rows:
        est_dt = estimate_posted_datetime(p.get("scraped_at"), p.get("posted_at") or "")
        p["_estimated_posted_at"] = est_dt
        if est_dt >= cutoff:
            recent.append(p)
            if len(recent) >= max_posts:
                break
    return recent


def parse_x_posted_at(posted_at_str: str | None) -> datetime | None:
    """
    X posted_at is described as a specific timestamp in your DB.
    Expect ISO-like strings; handle 'Z'.
    """
    if not posted_at_str:
        return None
    try:
        s = posted_at_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def fetch_recent_x_posts(
    supabase: Client,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    max_posts: int = MAX_POSTS_DEFAULT,
) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    # broad window by scraped_at (optional, but avoids scanning everything)
    scrape_start = now - timedelta(days=SCRAPE_WINDOW_DAYS)

    res = (
        supabase.table("x_posts")
        .select("id, full_text, posted_at, scraped_at, x_url, author_id, x_authors(display_name, x_handle)")
        .gte("scraped_at", scrape_start.isoformat())
        .lte("scraped_at", now.isoformat())
        .order("scraped_at", desc=True)
        .execute()
    )
    rows = res.data or []

    recent: List[Dict[str, Any]] = []
    for p in rows:
        posted_dt = parse_x_posted_at(p.get("posted_at"))
        if not posted_dt:
            # if posted_at missing/unparseable, fall back to scraped_at
            try:
                s = (p.get("scraped_at") or "").replace("Z", "+00:00")
                posted_dt = datetime.fromisoformat(s)
                if posted_dt.tzinfo is None:
                    posted_dt = posted_dt.replace(tzinfo=timezone.utc)
            except Exception:
                posted_dt = now

        p["_posted_dt"] = posted_dt
        if posted_dt >= cutoff:
            recent.append(p)
            if len(recent) >= max_posts:
                break
    return recent


# ---------- Build source text for the LLM ---------- #

def build_newsletter_source(linkedin_posts: List[Dict[str, Any]], x_posts: List[Dict[str, Any]]) -> str:
    """
    Produce a compact, LLM-friendly source bundle with both platforms.
    Keep it readable and consistent.
    """
    lines: List[str] = []

    lines.append("=== LINKEDIN POSTS ===")
    for p in linkedin_posts:
        author = (p.get("authors") or {}).get("display_name") or "Unknown"
        est = p.get("_estimated_posted_at")
        est_iso = est.isoformat() if est else "unknown"
        url = p.get("linkedin_url") or "no url"
        text = (p.get("full_text") or "").strip()
        if len(text) > 1200:
            text = text[:1200] + " [...]"
        lines.append(f"- Author: {author}")
        lines.append(f"  Date (estimated): {est_iso}")
        lines.append(f"  URL: {url}")
        lines.append(f"  Text: {text}")
        lines.append("")

    lines.append("=== X POSTS ===")
    for p in x_posts:
        author_info = p.get("x_authors") or {}
        author = author_info.get("display_name") or "Unknown"
        handle = author_info.get("x_handle") or ""
        posted_dt = p.get("_posted_dt")
        posted_iso = posted_dt.isoformat() if posted_dt else "unknown"
        url = p.get("x_url") or "no url"
        text = (p.get("full_text") or "").strip()
        if len(text) > 1200:
            text = text[:1200] + " [...]"
        lines.append(f"- Author: {author} @{handle}".strip())
        lines.append(f"  Date: {posted_iso}")
        lines.append(f"  URL: {url}")
        lines.append(f"  Text: {text}")
        lines.append("")

    return "\n".join(lines)


# ---------- HTML newsletter prompt ---------- #

NEWSLETTER_SYSTEM_PROMPT_HTML = """
Youre an editoraditorial assistant producing a readable weekly AI newsletter for busy executives.

You will be given recent LinkedIn and X posts (author, date, url, text).
Your job: synthesize 3–6 themes, cite notable posts, and provide actionable takeaways.

OUTPUT FORMAT RULES (STRICT):
- OUTPUT MUST BE VALID, EMAIL-SAFE HTML (not Markdown, not plain text).
- Use only these tags: html, body, h1, h2, h3, p, ul, li, strong, em, a, hr, br.
- No tables. No divs. No external CSS. No scripts. No images.
- Use minimal inline styling only on body and headings, suitable for email clients.
- Include links as <a href="...">source</a> where appropriate.

STYLE GOALS:
- Clean hierarchy, generous whitespace.
- Short paragraphs.
- Bullets for lists.
- Use <strong> sparingly for emphasis.
- Use <em> sparingly for tone.

STRUCTURE (use this order):
1) H1 title line: "AI WEEKLY — {Issue Date}"
2) One short tagline paragraph (emphasized).
3) "IN THIS ISSUE" section with 3–6 bullets.
4) "THIS WEEK’S THEMES" section: 3–5 themes. Each theme has:
   - What happened (bullets)
   - Why it matters (bullets)
   - Who said it (bullets with attribution + a link)
5) "NOTABLE POSTS" quick hits: 6–12 bullets max.

CONTENT CONSTRAINTS:
- Don’t invent facts. Only summarize what’s supported by the posts.
- Attribute claims when possible.
- Prefer synthesis over listing.
- Aim for ~700–1100 words.
"""


def generate_newsletter_html(source_text: str) -> str:
    issue_date = datetime.now().strftime("%b %d, %Y")
    user_prompt = (
        f"Issue Date: {issue_date}\n\n"
        f"Here are the recent posts:\n\n{source_text}"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.6,
        messages=[
            {"role": "system", "content": NEWSLETTER_SYSTEM_PROMPT_HTML},
            {"role": "user", "content": user_prompt},
        ],
    )
    html = (resp.choices[0].message.content or "").strip()
    return html


def html_to_plain_fallback(html: str) -> str:
    """
    Very simple fallback: strip tags crudely.
    Keeps automation easy and improves deliverability.
    """
    # remove tags
    text = re.sub(r"<[^>]+>", "", html)
    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or "AI Weekly Newsletter"


def derive_subject_from_html(html: str) -> str:
    """
    Try to pull H1 content; fallback to default.
    """
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        raw = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if raw:
            return raw[:200]
    return "AI Weekly Newsletter"


def parse_recipient_list(env_value: str, cli_value: str | None) -> List[str]:
    recipients: List[str] = []
    if env_value:
        recipients.extend([r.strip() for r in env_value.split(",") if r.strip()])
    if cli_value:
        recipients.extend([r.strip() for r in cli_value.split(",") if r.strip()])
    seen = set()
    unique = []
    for r in recipients:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def send_newsletter_email_html(
    subject: str,
    html_body: str,
    recipients: List[str],
):
    """
    Send multipart/alternative: plain text + HTML.
    """
    if not recipients:
        raise ValueError("No recipients provided for newsletter email.")
    if not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("SMTP_USER and SMTP_PASS must be set to send email.")

    plain_fallback = html_to_plain_fallback(html_body)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr((NEWSLETTER_FROM_NAME, SMTP_USER))
    msg["To"] = ", ".join(recipients)

    part1 = MIMEText(plain_fallback, "plain", "utf-8")
    part2 = MIMEText(html_body, "html", "utf-8")

    msg.attach(part1)
    msg.attach(part2)

    print(f"Sending newsletter email to: {', '.join(recipients)}")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, recipients, msg.as_string())

    print("Email sent successfully.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a styled weekly AI newsletter from recent LinkedIn + X posts.")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS_DEFAULT)
    parser.add_argument("--max-posts", type=int, default=MAX_POSTS_DEFAULT)
    parser.add_argument("--output-html", type=str, default="newsletter.html")
    parser.add_argument("--email-to", type=str, default="")

    args = parser.parse_args()

    supabase = get_supabase_client()

    linkedin_posts = fetch_recent_linkedin_posts(supabase, lookback_days=args.lookback_days, max_posts=args.max_posts)
    x_posts = fetch_recent_x_posts(supabase, lookback_days=args.lookback_days, max_posts=args.max_posts)

    print(f"Fetched {len(linkedin_posts)} recent LinkedIn posts.")
    print(f"Fetched {len(x_posts)} recent X posts.")

    source_text = build_newsletter_source(linkedin_posts, x_posts)

    html = generate_newsletter_html(source_text)

    # Basic sanity check
    if "<h1" not in html.lower():
        print("WARNING: Generated HTML does not contain an <h1>. Sending anyway, but consider tightening the prompt.")
    if "<script" in html.lower():
        raise RuntimeError("Unsafe HTML: script tag detected.")

    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nNewsletter HTML written to {args.output_html}\n")

    recipients = parse_recipient_list(NEWSLETTER_RECIPIENTS_ENV, args.email_to)
    if not recipients:
        print("ERROR: No recipients found. Set NEWSLETTER_RECIPIENTS or pass --email-to.")
        return

    subject = derive_subject_from_html(html)
    send_newsletter_email_html(subject, html, recipients)


if __name__ == "__main__":
    main()