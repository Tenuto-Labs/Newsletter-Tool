# login_and_save.py
import asyncio
import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

STATE_PATH = os.environ.get("STATE_PATH", "state.json")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # Go to LinkedIn login page
        await page.goto("https://www.linkedin.com/login")

        print(
            "\n== LinkedIn login ==\n"
            "1. Log in manually in the browser window.\n"
            "2. After login completes and your feed loads, press ENTER here.\n"
        )

        # Wait for you to complete login
        input("Press ENTER here when you’re logged in in the browser...")

        # Save cookies & session
        await context.storage_state(path=STATE_PATH)
        print(f"Saved storage state to {STATE_PATH}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
