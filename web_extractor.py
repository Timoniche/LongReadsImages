"""
Web page text extraction
"""

from urllib.parse import urlparse
from pathlib import Path
from typing import Optional

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import trafilatura

    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def extract_text(html: str, url: str = "") -> str:
    """Extract text from HTML using trafilatura"""
    if not HAS_TRAFILATURA:
        raise ImportError(
            "trafilatura is required. Install it with: pip install trafilatura"
        )

    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False
    )
    if not text:
        raise ValueError("trafilatura failed to extract text from the page")
    return text.strip()


def fetch_with_playwright(url: str, wait_time: int = 5000, timeout: int = 60000) -> str:
    """
    Fetch a web page using Playwright (handles JavaScript-rendered pages).

    Args:
        url: URL to fetch
        wait_time: Time to wait for page to load (milliseconds)
        timeout: Maximum timeout for page navigation (milliseconds)

    Returns:
        HTML content after JavaScript execution
    """
    if not HAS_PLAYWRIGHT:
        raise ImportError(
            "Playwright is required for JavaScript-rendered pages. "
            "Install it with: pip install playwright && playwright install chromium"
        )

    print("Fetching with Playwright (JavaScript rendering)...")

    with sync_playwright() as p:
        # Launch browser (headless)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        try:
            try:
                print("   Waiting for page to load...")
                page.goto(url, wait_until="load", timeout=timeout)
            except (TimeoutError, RuntimeError):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                except (TimeoutError, RuntimeError):
                    page.goto(url, wait_until="commit", timeout=timeout)

            print(f"   Waiting {wait_time}ms for content to render...")
            page.wait_for_timeout(wait_time)

            try:
                page.wait_for_selector("body", timeout=5000)
            except (TimeoutError, RuntimeError):
                pass

            # Get HTML after JavaScript execution
            html = page.content()
            browser.close()

            print(f"Fetched {len(html):,} bytes (after JS execution)")
            return html

        except (TimeoutError, RuntimeError, ValueError) as e:
            browser.close()
            raise RuntimeError(f"Playwright failed to fetch page: {e}")


def fetch_and_extract(url: str, save_to: Optional[Path] = None) -> str:
    """
    Fetch a web page and extract its text content.
    Tries static HTML first, then Playwright if that fails.

    Args:
        url: URL to fetch
        save_to: Optional path to save extracted text

    Returns:
        Extracted text content
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "'requests' library is required. Install it with: pip install requests"
        )

    # Try static HTML first
    try:
        print(f"Fetching: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        html = response.text
        print(f"Fetched {len(html):,} bytes")

        # Extract text
        print("Extracting text...")
        text = extract_text(html, url)

        if len(text) < 50:
            raise ValueError(f"Extracted text too short ({len(text)} chars)")

        print(f"Extracted {len(text):,} characters")

    except (ValueError, requests.RequestException) as e:
        # If static extraction failed, try Playwright
        if HAS_PLAYWRIGHT:
            print(f"Static extraction failed: {e}")
            print("Retrying with Playwright (JavaScript rendering)...")

            html = fetch_with_playwright(url)
            print("Extracting text...")
            text = extract_text(html, url)

            if len(text) < 50:
                raise ValueError(
                    f"Extracted text too short ({len(text)} chars) even after JavaScript rendering"
                )

            print(f"Extracted {len(text):,} characters")
        else:
            raise RuntimeError(
                f"Static extraction failed and Playwright is not available. Error: {e}"
            )

    # Save to file if requested
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(text, encoding="utf-8")
        print(f"Saved to: {save_to}")

    return text


def is_url(text: str) -> bool:
    """Check if a string is a URL"""
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except (ValueError, AttributeError, TypeError):
        return False
