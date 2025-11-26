import asyncio
import os
import hashlib
from dataclasses import dataclass
from typing import List, Optional

import requests
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
STATE_PATH = os.environ.get("STATE_PATH", "state.json")

# Buckets you created in Supabase Storage
RAW_HTML_BUCKET = "raw-html"
IMAGES_BUCKET = "post-images"

# Ethan's profile/posts URLs
ETHAN_PROFILE_URL = "https://www.linkedin.com/in/emollick/"
ETHAN_POSTS_URL = "https://www.linkedin.com/in/emollick/recent-activity/all/"


@dataclass
class ScrapedPost:
    linkedin_post_id: str
    linkedin_url: str
    posted_at: Optional[str]
    full_text: str
    html: str
    image_urls: List[str]


# ---------- Supabase helpers ---------- #
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def upsert_author(supabase: Client, profile_url: str, display_name: str) -> int:
    """
    Upsert an author row and return its id.
    """
    res = (
        supabase.table("authors")
        .upsert(
            {
                "linkedin_profile_url": profile_url,
                "display_name": display_name,
            },
            on_conflict="linkedin_profile_url",
        )
        .execute()
    )
    # Supabase returns list of rows
    author_id = res.data[0]["id"]
    return author_id


def get_post_by_linkedin_id(supabase: Client, linkedin_post_id: str):
    res = (
        supabase.table("posts")
        .select("id")
        .eq("linkedin_post_id", linkedin_post_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def upload_html(supabase: Client, post_id: str, html: str) -> str:
    """
    Upload HTML to the raw-html bucket, return storage path.
    If the object already exists, overwrite it (upsert).
    """
    path = f"{post_id}/raw.html"
    supabase.storage.from_(RAW_HTML_BUCKET).upload(
        path,
        html.encode("utf-8"),
        {"upsert": "true"},
    )
    return path


def download_and_upload_image(
    supabase: Client, post_id: str, idx: int, url: str
) -> str:
    """
    Download image from LinkedIn CDN and upload to Supabase Storage.
    If the image already exists at that path, overwrite it.
    """
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    content = resp.content

    path = f"{post_id}/{idx}.jpg"
    supabase.storage.from_(IMAGES_BUCKET).upload(
        path,
        content,
        {"upsert": "true"},
    )
    return path


def insert_post(
    supabase: Client,
    author_id: int,
    scraped_post: ScrapedPost,
    html_storage_path: str,
):
    res = (
        supabase.table("posts")
        .insert(
            {
                "author_id": author_id,
                "linkedin_post_id": scraped_post.linkedin_post_id,
                "linkedin_url": scraped_post.linkedin_url,
                "posted_at": scraped_post.posted_at,  # raw string; can normalize later
                "full_text": scraped_post.full_text,
                "raw_html_storage_path": html_storage_path,
            }
        )
        .execute()
    )
    return res.data[0]["id"]  # UUID


def insert_post_image(
    supabase: Client,
    post_id: str,
    storage_path: str,
    original_src_url: str,
    position: int,
):
    supabase.table("post_images").insert(
        {
            "post_id": post_id,
            "storage_path": storage_path,
            "original_src_url": original_src_url,
            "position": position,
        }
    ).execute()


# ---------- Scraper (Playwright) ---------- #


async def scrape_ethan_posts(max_scrolls: int = 40) -> List[ScrapedPost]:
    """
    Scrape Ethan Mollick's posts from his recent activity page.

    Strategy:
      - Treat each 'feed-shared-update-v2' card as a post container.
      - Inside each card:
          - Text: first div.update-components-text
          - Date: first <time> element (use datetime attr if present),
                  else span whose class contains 'update-components-actor__sub-description'
          - Images: img.update-components-image__image,
                    or fallback to img[src*='media.licdn.com']
    """
    posts: dict[str, ScrapedPost] = {}
    no_new_posts_scrolls = 0  # heuristic to stop when feed stops changing

    async with async_playwright() as p:
        # headless=False so you can see what's happening while debugging
        browser = await p.chromium.launch(headless=False, slow_mo=250)
        context = await browser.new_context(storage_state=STATE_PATH)
        page = await context.new_page()

        await page.goto(ETHAN_POSTS_URL)
        await page.wait_for_timeout(5000)
        print("Current URL:", page.url)

        # Quick sanity check: are we accidentally on a login page?
        if await page.locator("input[name='session_key']").count() > 0:
            print("Looks like the LinkedIn login page — your saved state may be expired.")
            print("Re-run login_and_save_state.py to refresh state.json.")
            await browser.close()
            return []

        for scroll_idx in range(max_scrolls):
            # Scroll to load more posts
            await page.mouse.wheel(0, 2500)
            await page.wait_for_timeout(2000)

            # Each full card for an update / post
            post_locators = page.locator("div.feed-shared-update-v2")
            card_count = await post_locators.count()
            print(f"[scroll {scroll_idx}] found {card_count} post cards so far")

            new_posts_this_scroll = 0

            for idx in range(card_count):
                root = post_locators.nth(idx)

                # ----- Extract main text -----
                commentary = root.locator("div.update-components-text").first
                if await commentary.count() == 0:
                    continue

                text = (await commentary.inner_text()).strip()
                if not text:
                    continue

                # Synthesize a stable-ish ID from the text content
                hash_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                linkedin_post_id = hash_id

                if linkedin_post_id in posts:
                    continue  # already captured this post

                html = await root.inner_html()

                # ----- Date / posted_at -----
                posted_at: Optional[str] = None

                # 1) Try <time> element
                time_elem = root.locator("time").first
                if await time_elem.count() > 0:
                    dt_attr = await time_elem.get_attribute("datetime")
                    if dt_attr:
                        posted_at = dt_attr.strip()
                    else:
                        inner = (await time_elem.inner_text() or "").strip()
                        posted_at = inner or None
                else:
                    # 2) Fallback: sub-description span (often contains "1d", "3w", etc.)
                    sub_desc = root.locator(
                        "span.update-components-actor__sub-description"
                    ).first
                    if await sub_desc.count() > 0:
                        txt = (await sub_desc.inner_text() or "").strip()
                        posted_at = txt or None

                # ----- Images -----
                image_urls: List[str] = []

                # Primary: specific post image class
                image_locator = root.locator("img.update-components-image__image")
                img_count = await image_locator.count()

                # Fallback: any LinkedIn media image under this root
                if img_count == 0:
                    image_locator = root.locator("img[src*='media.licdn.com']")
                    img_count = await image_locator.count()

                for j in range(img_count):
                    src = await image_locator.nth(j).get_attribute("src")
                    if src:
                        image_urls.append(src)

                # Deduplicate
                image_urls = list(dict.fromkeys(image_urls))

                posts[linkedin_post_id] = ScrapedPost(
                    linkedin_post_id=linkedin_post_id,
                    linkedin_url="",  # permalink unknown for now
                    posted_at=posted_at,   # may be None if no suitable element
                    full_text=text,
                    html=html,
                    image_urls=image_urls,
                )

                new_posts_this_scroll += 1

            print(
                f"[scroll {scroll_idx}] new posts found this scroll: {new_posts_this_scroll}"
            )

            if new_posts_this_scroll == 0:
                no_new_posts_scrolls += 1
            else:
                no_new_posts_scrolls = 0

            # Heuristic: if we've scrolled several times with no new posts, stop early
            if no_new_posts_scrolls >= 3:
                print(
                    "No new posts found in the last 3 scrolls. Assuming we've reached the end of the feed."
                )
                break

        await browser.close()

    print(f"Total unique posts scraped: {len(posts)}")
    return list(posts.values())


# ---------- Orchestration ---------- #


async def main():
    print("Scraping Ethan's posts from LinkedIn...")
    scraped_posts = await scrape_ethan_posts(max_scrolls=40)
    print(f"Scraped {len(scraped_posts)} posts")

    supabase = get_supabase_client()

    # Upsert Ethan as an author
    author_id = upsert_author(
        supabase,
        profile_url=ETHAN_PROFILE_URL,
        display_name="Ethan Mollick",
    )
    print(f"Author id: {author_id}")

    # Store each post if not already present
    created_count = 0

    for sp in scraped_posts:
        existing = get_post_by_linkedin_id(supabase, sp.linkedin_post_id)
        if existing:
            print(f"Skipping existing post {sp.linkedin_post_id}")
            continue

        print(f"Storing new post {sp.linkedin_post_id}")

        # 1) Upload HTML
        html_storage_path = upload_html(
            supabase, sp.linkedin_post_id, sp.html
        )

        # 2) Insert post row
        post_id = insert_post(
            supabase,
            author_id=author_id,
            scraped_post=sp,
            html_storage_path=html_storage_path,
        )

        # 3) Upload images and create post_images rows
        for idx, image_url in enumerate(sp.image_urls):
            try:
                image_path = download_and_upload_image(
                    supabase, sp.linkedin_post_id, idx, image_url
                )
                insert_post_image(
                    supabase,
                    post_id=post_id,
                    storage_path=image_path,
                    original_src_url=image_url,
                    position=idx,
                )
            except Exception as e:
                print(f"Failed to store image {image_url}: {e}")

        created_count += 1

    print(f"Done. Created {created_count} new posts in Supabase.")


if __name__ == "__main__":
    asyncio.run(main())