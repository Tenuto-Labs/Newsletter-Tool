import asyncio
import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

STATE_PATH = os.environ.get("X_STATE_PATH", "x_state.json")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=150)
        context = await browser.new_context()
        page = await context.new_page()

        # Go to X login
        await page.goto("https://x.com/i/flow/login")
        print("\n1) Log into X in the opened browser window (handle any 2FA).")
        print("2) Once you're fully logged in and can see your home feed, come back here.")
        input("\nPress ENTER here after login is complete to save state.json... ")

        await context.storage_state(path=STATE_PATH)
        print(f"\nSaved storage state to: {STATE_PATH}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())