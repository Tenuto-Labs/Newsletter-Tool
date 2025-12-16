import asyncio
import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict
from urllib.parse import urlparse

from dotenv import load_dotenv
from playwright.async_api import async_playwright
from supabase import create_client, Client

load_dotenv()

# -------------------- ENV --------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
X_STATE_PATH = os.environ.get("X_STATE_PATH", "x_state.json")

# Optional HTML storage (uses your existing bucket)
STORE_RAW_HTML = True
RAW_HTML_BUCKET = "raw-html"

# -------------------- TABLES --------------------
X_AUTHORS_TABLE = "x_authors"
X_POSTS_TABLE = "x_posts"

# -------------------- AUTHORS --------------------
AUTHORS = [
    {"display_name": "Ethan Mollick", "profile_url": "https://x.com/emollick?lang=en"},
    {"display_name": "Andrew Ng", "profile_url": "https://x.com/AndrewYNg"},
    {"display_name": "Yann LeCun", "profile_url": "https://x.com/ylecun?lang=en"},
    {"display_name": "Mustafa Suleyman", "profile_url": "https://x.com/mustafasuleyman?lang=en"},
    {"display_name": "Demis Hassabis", "profile_url": "https://x.com/demishassabis?lang=en"},
    {"display_name": "Fei-Fei Li", "profile_url": "https://x.com/drfeifei?lang=en"},
    {"display_name": "Yoshua Bengio", "profile_url": "https://x.com/Yoshua_Bengio"},
]

STATUS_RE = re.compile(r"/status/(\d+)")


@dataclass
class XScrapedPost:
    x_post_id: str
    x_url: str
    posted_at: Optional[str]
    full_text: str
    html: str


# -------------------- HELPERS --------------------
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def canonical_profile_url(url: str) -> str:
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}{u.path}".rstrip("/")


def extract_handle(profile_url: str) -> str:
    u = urlparse(profile_url)
    path = (u.path or "").strip("/")
    return path.split("/")[0] if path else ""


def upsert_x_author(supabase: Client, profile_url: str, display_name: str) -> int:
    handle = extract_handle(profile_url)
    if not handle:
        raise ValueError(f"Could not extract x_handle from profile_url: {profile_url}")

    res = (
        supabase.table(X_AUTHORS_TABLE)
        .upsert(
            {"x_profile_url": profile_url, "x_handle": handle, "display_name": display_name},
            on_conflict="x_profile_url",
        )
        .execute()
    )
    return res.data[0]["id"]


def get_post_by_x_id(supabase: Client, x_post_id: str):
    res = (
        supabase.table(X_POSTS_TABLE)
        .select("id")
        .eq("x_post_id", x_post_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def upload_html(supabase: Client, x_post_id: str, html: str) -> Optional[str]:
    if not STORE_RAW_HTML:
        return None
    path = f"x/{x_post_id}/raw.html"
    supabase.storage.from_(RAW_HTML_BUCKET).upload(
        path,
        html.encode("utf-8"),
        {"upsert": "true"},
    )
    return path


def insert_x_post(supabase: Client, author_id: int, sp: XScrapedPost, raw_html_storage_path: Optional[str]):
    res = (
        supabase.table(X_POSTS_TABLE)
        .insert(
            {
                "author_id": author_id,
                "x_post_id": sp.x_post_id,
                "x_url": sp.x_url,
                "posted_at": sp.posted_at,
                "full_text": sp.full_text,
                "raw_html_storage_path": raw_html_storage_path,
            }
        )
        .execute()
    )
    return res.data[0]["id"]


async def ensure_logged_in_and_save_state(page, context):
    # If redirected to login, let user log in manually, then save state
    if "flow/login" in page.url or "/login" in page.url:
        print("\n⚠️ Redirected to login. Please log in in the browser window.")
        print("Once you reach your feed or the profile page, come back here.\n")

        for _ in range(600):  # 10 min max
            await page.wait_for_timeout(1000)
            if "flow/login" not in page.url and "/login" not in page.url:
                break

        await context.storage_state(path=X_STATE_PATH)
        print(f"✅ Saved login state to {X_STATE_PATH}\n")


# -------------------- SCRAPER --------------------
async def scrape_x_profile_posts(profile_url: str, max_scrolls: int = 40) -> List[XScrapedPost]:
    posts: Dict[str, XScrapedPost] = {}
    no_new_scrolls = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=50)

        # ✅ FIX: new_context() must be on the browser, not p.chromium
        context = await browser.new_context(storage_state=X_STATE_PATH)

        page = await context.new_page()

        print(f"Navigating to: {profile_url}")
        await page.goto(profile_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(4000)

        await ensure_logged_in_and_save_state(page, context)

        # If login flow returned us somewhere else, go back to the profile
        if canonical_profile_url(page.url) != canonical_profile_url(profile_url):
            await page.goto(profile_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

        for scroll_idx in range(max_scrolls):
            articles = page.locator("article[data-testid='tweet']")
            count = await articles.count()
            print(f"[scroll {scroll_idx}] tweet articles visible: {count}")

            new_this_scroll = 0

            for i in range(count):
                a = articles.nth(i)

                time_el = a.locator("time").first
                if await time_el.count() == 0:
                    continue

                posted_at = await time_el.get_attribute("datetime")

                # Prefer the permalink that wraps the time element
                time_link = time_el.locator("xpath=ancestor::a[1]").first
                href = await time_link.get_attribute("href")
                if not href:
                    continue

                m = STATUS_RE.search(href)
                if not m:
                    continue

                x_post_id = m.group(1)
                if x_post_id in posts:
                    continue

                text_el = a.locator("[data-testid='tweetText']").first
                if await text_el.count() == 0:
                    continue

                text = (await text_el.inner_text()).strip()
                if not text:
                    continue

                html = await a.inner_html()

                posts[x_post_id] = XScrapedPost(
                    x_post_id=x_post_id,
                    x_url=f"https://x.com{href}",
                    posted_at=posted_at,
                    full_text=text,
                    html=html,
                )
                new_this_scroll += 1

            print(f"[scroll {scroll_idx}] new posts captured: {new_this_scroll}")

            if new_this_scroll == 0:
                no_new_scrolls += 1
            else:
                no_new_scrolls = 0

            if no_new_scrolls >= 3:
                print("No new posts in last 3 scrolls — stopping early.")
                break

            await page.mouse.wheel(0, 2500)
            await page.wait_for_timeout(2500)

        await browser.close()

    print(f"Total unique posts scraped from {canonical_profile_url(profile_url)}: {len(posts)}")
    return list(posts.values())


# -------------------- PIPELINE --------------------
async def run_for_author(display_name: str, profile_url: str, max_scrolls: int = 40):
    print("=" * 80)
    print(f"Scraping X posts for: {display_name}")
    print(f"Profile URL: {profile_url}")
    print("=" * 80)

    scraped = await scrape_x_profile_posts(profile_url=profile_url, max_scrolls=max_scrolls)
    print(f"Scraped {len(scraped)} posts for {display_name}")

    if not scraped:
        print("No posts scraped — skipping storage.")
        return

    supabase = get_supabase_client()

    author_id = upsert_x_author(
        supabase,
        profile_url=canonical_profile_url(profile_url),
        display_name=display_name,
    )
    print(f"Author id: {author_id}")

    created = 0
    skipped = 0

    for sp in scraped:
        if get_post_by_x_id(supabase, sp.x_post_id):
            skipped += 1
            continue

        raw_path = upload_html(supabase, sp.x_post_id, sp.html) if STORE_RAW_HTML else None
        insert_x_post(supabase, author_id, sp, raw_path)
        created += 1

    print(f"Done for {display_name}. Created={created}, Skipped existing={skipped}")


async def main():
    parser = argparse.ArgumentParser(description="Scrape X posts and store in Supabase (text-only).")
    parser.add_argument("--only", type=str, default=None, help="Only scrape this display name (exact match).")
    parser.add_argument("--max_scrolls", type=int, default=40, help="Max scroll iterations per author.")
    args = parser.parse_args()

    authors_to_run = AUTHORS
    if args.only:
        authors_to_run = [a for a in AUTHORS if a["display_name"] == args.only]
        if not authors_to_run:
            print(f"No author with display name '{args.only}' found.")
            return

    for a in authors_to_run:
        await run_for_author(a["display_name"], a["profile_url"], max_scrolls=args.max_scrolls)

    print("All requested authors processed.")


if __name__ == "__main__":
    asyncio.run(main())