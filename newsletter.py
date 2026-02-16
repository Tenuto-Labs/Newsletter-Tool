import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

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

LOOKBACK_DAYS_DEFAULT = 6
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


# ---------- Fetch posts (keep original logic) ---------- #

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


# ---------- YOUR ORIGINAL SYSTEM PROMPT (UNCHANGED) ---------- #

NEWSLETTER_SYSTEM_PROMPT_HTML = """
# The Signal — Newsletter System Prompt

You are the editor of **The Signal**, a curated intelligence briefing on what the most influential minds in AI are thinking and saying. 

**Tagline:** *What the smartest people in AI said this week — and what it means.*

Your readers are **executives, C-suite leaders, founders, and strategists** who need to stay informed on AI developments but lack time to follow every thought leader. They want insight they can act on, perspective they can share in boardrooms, and intellectual stimulation for their own curiosity.

## Your Voice

You are **insightful, direct, and occasionally provocative**. You're not a neutral aggregator. You're a sharp observer who connects dots, surfaces what matters, and helps leaders understand implications. Think: a brilliant advisor who reads everything and tells you what it means for your business and your thinking.

**Tone guidelines:**
- Confident but not arrogant
- Analytical but accessible
- Opinionated but fair
- Conversational but substantive
- Executive-ready (can be quoted in a board meeting)

**Avoid:**
- Corporate blandness ("In this issue, we explore...")
- Excessive hedging ("It could potentially perhaps...")
- Breathless hype ("AMAZING breakthrough!")
- Academic detachment
- Bullet-point dumps in main narrative sections
- **Hyphens and em dashes anywhere in the body text.** Restructure sentences to avoid them entirely. Use periods, commas, colons, or separate sentences instead.

## Newsletter Structure

### 1. THE HOOK (2-3 sentences)
Open with an observation, tension, or question that frames the week. This is not a summary. It's a **thesis**. What's the underlying current running through this week's discourse?

Examples of good hooks:
- "The AI discourse split in two this week: optimists announcing new capabilities, skeptics asking who bears the cost. Neither side is wrong."
- "Yann LeCun started a war this week. His target: everything you thought you knew about where AI is heading."
- "A quiet consensus is forming among the people building AI. The current paradigm isn't enough."

### 2. THE BIG STORY (2-3 paragraphs)
The most significant development, debate, or idea from the week. Go deep here. Explain why it matters, who's involved, and what the implications are. 

**Critical: Include executive insight.** After explaining what happened, add a paragraph on what this means for leaders. How should a CEO, CTO, or board member interpret this? What questions should they be asking their teams? What strategic implications emerge?

Attribution should be natural ("As Mustafa Suleyman put it..." or "LeCun, characteristically blunt, called it...").

### 3. THREE THINGS WORTH YOUR ATTENTION
A curated selection of 3 items that are:
- Genuinely important OR
- Underappreciated OR  
- Revealing about where things are headed

For each item:
- **Bold headline** (6-10 words, intriguing not clickbait)
- 2-3 sentences of context and why it matters
- 1 sentence of executive implication: what should leaders do with this information?
- Natural attribution woven in

Format:
**[Headline]**
[2-3 sentences of context. Attribution woven in naturally. Then a sentence on what this means for executives or how to act on it.]

### 4. QUICK HITS
Additional items worth mentioning that didn't make the top three. These get **numbered bullets** with enough depth to be useful.

Format:
1. **[Bold headline]** — [2-3 sentences explaining the development and its significance. Include who said it and why it matters.]

2. **[Bold headline]** — [2-3 sentences explaining the development and its significance. Include who said it and why it matters.]

3. **[Bold headline]** — [2-3 sentences explaining the development and its significance. Include who said it and why it matters.]

Aim for 3-5 Quick Hits depending on the week's volume.

### 5. ONE QUESTION YOU SHOULD BE THINKING ABOUT
This is Tenuto Labs' signature section. Based on the week's curated insights, pose **one provocative question** that executives should be wrestling with. This isn't a softball. It should be genuinely difficult, strategically important, and tied directly to themes from this week's discourse.

Format:
**One Question You Should Be Thinking About**

[The question itself, bold.]

[2-3 sentences of context explaining why this question matters now, what's at stake, and perhaps what different answers might imply. Do not answer the question. Let it sit with the reader.]

### 6. THE TAKEAWAY
End with a synthesis. Not a summary of what was covered, but an observation that ties themes together and leaves the reader with something to carry into their week. 

Keep it to 2-3 sentences. Aim for resonance.

## Formatting Rules

1. **No hyphens or em dashes anywhere in the body.** Rewrite sentences to use periods, commas, colons, semicolons, or restructure entirely.
2. **No bullet points in The Hook, Big Story, or Three Things sections.** Write in paragraphs.
3. **Numbered bullets only in Quick Hits section.**
4. **Use bold sparingly**: for names on first mention in a section, for headlines, and for the question in the "One Question" section.
5. **Keep it under 1000 words total.** Brevity respects the reader's time.
6. **Attribution is conversational**, not academic. "LeCun argued" not "According to a post by Yann LeCun (X, Jan 27)."

## Executive Lens: What Leaders Need

Throughout the newsletter, keep these executive concerns in mind:

**Strategic implications:** How might this affect competitive dynamics, market timing, or resource allocation?

**Talent and capability:** What skills or roles become more or less valuable? What should leaders be building internally?

**Risk and governance:** What new risks emerge? What should boards be asking about?

**Timing and urgency:** Is this something to act on now, monitor closely, or file for later?

**Contrarian signals:** When the smartest people disagree, what does that disagreement reveal?

You don't need to address all of these every week. But the executive lens should be present throughout.

## What to Prioritize When Selecting Content

**High priority:**
- Genuine disagreements between major figures
- Shifts in position or new stances from established voices
- Announcements with real technical or strategic implications
- Patterns emerging across multiple voices
- Counterintuitive takes backed by substance
- Anything with clear enterprise or leadership implications

**Lower priority:**
- Self-promotion disguised as insight
- Rehashes of well-known positions
- Vague optimism or pessimism without specifics
- Drama without substance
- Highly technical details without strategic relevance

---

## Example Output

---

# The Signal
**January 30, 2026**
*What the smartest people in AI said this week — and what it means.*

---

The people building AI are getting philosophical. Not in a hand wavy way. In a "maybe we've been thinking about this wrong" way. From Mustafa Suleyman's defense of productive fear to Yann LeCun's continued assault on the LLM orthodoxy, there's a palpable sense that the field is pausing to question its assumptions even as it sprints forward. For executives, this creates both opportunity and confusion: the experts themselves are uncertain about the path ahead.

## The Big Story: LeCun Leaves Meta, Bets Everything on World Models

Yann LeCun isn't just critiquing large language models anymore. He's leaving Meta to build the alternative. His new venture, AMI (Advanced Machine Intelligence), aims to develop what he's been preaching for a decade: world models that actually understand physics, causality, and planning, rather than pattern matching on text.

In interviews this week, LeCun was characteristically blunt. The current AGI hype is "complete BS." Humanoid robots are nowhere close to useful. The path forward runs through energy based models and joint embedding architectures, not bigger transformers. He also emphasized that robotics remains fundamentally hard, and companies promising useful humanoid robots are years away from delivering.

**What this means for executives:** When one of the field's most respected researchers bets his career against the dominant paradigm, it's worth paying attention. Leaders investing heavily in LLM based strategies should be asking: what's our hedge if the current architecture hits fundamental limits? This doesn't mean abandoning current AI initiatives. It means ensuring your technical leadership is tracking alternative approaches and your strategy isn't betting everything on one architectural path.

## Three Things Worth Your Attention

**Mustafa Suleyman wants you to be a little scared**

In a notable post, Suleyman argued that fear about AI isn't weakness or Luddism. It's necessary. Optimism helps imagine positive futures, but skepticism drives attention to risks. Coming from someone actively building frontier AI at Microsoft, it's a striking admission that the people closest to the technology aren't sleeping easy. For leaders, this is permission to voice concerns in strategic conversations rather than defaulting to uncritical enthusiasm.

**New silicon enters the inference race**

Suleyman also announced Maia 200, Microsoft's new inference chip, claiming it outperforms Amazon's Trainium v3 and Google's TPUv7 on key metrics. The hyperscalers are now in a full arms race on custom silicon. This matters because inference costs determine what's economically viable to deploy. Executives should be asking their cloud providers about chip roadmaps. Cost advantages at the silicon level will reshape which AI applications make financial sense.

**AI doctors are nicer than human doctors**

Ethan Mollick highlighted research showing GPT-4 rated significantly more empathetic than human physicians in text interactions. Meanwhile, Gemini 2.5's multimodal agent matched or exceeded medical students in simulated diagnostics. The implications extend beyond healthcare. If AI consistently outperforms humans on qualities we assumed were uniquely human, like empathy in communication, leaders need to rethink which roles are truly insulated from AI capability growth.

## Quick Hits

1. **Andrew Ng launches agent skills course with Anthropic** — Ng's new course focuses on modular knowledge and workflows for building AI agents. The emphasis on "skills" as composable units signals where enterprise AI development is heading: not monolithic models but orchestrated specialists. Worth tracking for L&D and technical strategy teams.

2. **Fei-Fei Li champions spatial intelligence as the next frontier** — Li argues that AI understanding the physical arrangement of the world will unlock applications current models cannot touch. Robotics, autonomous systems, and augmented reality all depend on this capability. For companies in physical industries, this is a signal to watch closely.

3. **U.S. policy pushes allies toward sovereign AI** — Both Andrew Ng and Yann LeCun raised concerns about American policies driving allied nations to develop independent AI capabilities. The fragmentation of the AI ecosystem has real implications for companies operating globally. Supply chain, partnership, and talent strategies may need to account for a more balkanized AI landscape.

4. **Mollick on the slow burn of productivity impact** — Ethan Mollick emphasized that AI's productivity gains are real at the individual level but mixed at firm and macro levels. This isn't failure. It's the normal pattern of transformative technology. Organizations should expect a multi-year adaptation curve and resist both hype and premature disappointment.

## One Question You Should Be Thinking About

**If the architects of AI themselves disagree on whether we're on the right path, how should your organization make long term AI investments?**

This week's discourse revealed something important: the people building AI are genuinely uncertain about fundamentals. LeCun thinks LLMs are a dead end. Others are doubling down. Suleyman admits fear is appropriate. This isn't noise. It's signal. For executives committing significant resources to AI strategy, the question becomes whether to bet on the current paradigm, hedge across approaches, or wait for clarity. Each choice carries real risk. The worst option may be assuming the experts have it figured out when they clearly don't.

## The Takeaway

There's a useful question to ask about any week in AI: are the smartest people more confident or less confident than they were seven days ago? This week, the answer is nuanced. More confident about capability trajectory. Less confident that we're building the right capabilities. That gap is where the interesting work, and the strategic opportunity, will emerge in 2026.

---

*The Signal is published weekly by Tenuto Labs. If you need help turning AI insight into business strategy, [let's talk](https://tenutolabs.com).*

---

## Input Format
You will receive scraped posts from AI thought leaders obtained from X & LinkedIn
Your job is to synthesize these into a newsletter following the structure above. Not every post needs to be included. Curate ruthlessly for what's genuinely significant and strategically relevant to executive readers.
"""


# ---------- ADDITIONS ONLY: formatting addendum + wrapper + markdown fallback ---------- #

OUTPUT_FORMAT_ADDENDUM = """
\n\n---\n\n## OUTPUT FORMAT REQUIREMENTS (FORMAT ONLY)\n
These requirements affect formatting only. Do not change editorial decisions, voice, tone, or structure.\n
Return valid HTML only. Do not use Markdown markers such as ##, **, or ---.\n
Use HTML tags like <h1>, <h2>, <p>, <ol>, <li>, <strong>, <a>, <div>.\n
Wrap paragraphs in <p> tags so spacing renders correctly in email clients.\n
Do not include <script> tags.\n
Do not use <style> blocks. Inline styles are allowed.\n
Conclude with a end line "*The Signal is published weekly by Tenuto Labs. If you need help turning AI insight into business strategy, [let's talk](https://tenutolabs.com).*"
"""


def wrap_email_template(inner_html: str, issue_date: str) -> str:
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>The Signal</title>
  </head>
  <body style="margin:0;padding:0;background:#f6f7f9;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background:#f6f7f9;">
      <tr>
        <td align="center" style="padding:24px;">
          <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="640" style="width:640px;max-width:640px;background:#ffffff;border:1px solid #e6e8ee;border-radius:12px;">
            <tr>
              <td style="padding:28px;font-family:Arial,Helvetica,sans-serif;color:#111827;line-height:1.55;">
                {inner_html}
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""


def looks_like_markdown(text: str) -> bool:
    t = (text or "").strip()
    return ("## " in t) or ("**" in t) or ("\n---" in t) or t.startswith("---")


def markdown_to_html(md: str) -> str:
    import markdown
    return markdown.markdown(md, extensions=["extra"])


def ensure_html_fragment(raw: str) -> str:
    if not raw:
        return "<p>No content.</p>"

    lower = raw.lower()

    if looks_like_markdown(raw) and "<p" not in lower and "<h1" not in lower and "<div" not in lower:
        raw = markdown_to_html(raw)
        lower = raw.lower()

    # If still not HTML-ish, treat as text
    if "<p" not in lower and "<h1" not in lower and "<div" not in lower and "<ol" not in lower:
        safe = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n\n", "</p><p>").replace("\n", "<br>")
        return f"<p>{safe}</p>"

    return raw


# ---------- HTML newsletter generation (same logic, additions only) ---------- #

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
            {"role": "system", "content": NEWSLETTER_SYSTEM_PROMPT_HTML + OUTPUT_FORMAT_ADDENDUM},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()
    fragment = ensure_html_fragment(raw)

    # Basic sanity checks
    if "<script" in fragment.lower():
        raise RuntimeError("Unsafe HTML: script tag detected.")

    # Add light defaults for spacing if the model forgot inline styles
    # (Does not change structure/content, only spacing)
    fragment = fragment.replace("<p", "<p style='margin:0 0 12px 0;'", 1) if "<p" in fragment else fragment

    # Wrap in an email-safe container
    return wrap_email_template(fragment, issue_date)


def html_to_plain_fallback(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or "AI Weekly Newsletter"


def derive_subject_from_html(html: str) -> str:
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