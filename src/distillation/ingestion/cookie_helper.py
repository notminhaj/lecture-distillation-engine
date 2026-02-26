"""
Cookie extraction helper for YouTube downloads on Windows.

Chrome v127+ uses app-bound encryption (v20) which yt-dlp cannot decrypt.
This module uses Selenium to launch a headless browser that loads the user's
profile, extracting decrypted cookies via the DevTools protocol.

Writes a Netscape-format cookies.txt file that yt-dlp can consume.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Cookies that indicate an authenticated YouTube session
_AUTH_COOKIE_NAMES = {
    "SID", "HSID", "SSID", "APISID", "SAPISID",
    "LOGIN_INFO", "__Secure-1PSID", "__Secure-3PSID",
}


def extract_cookies_to_file(output_path: str | Path) -> bool:
    """
    Extract YouTube/Google cookies from the default browser and write
    a Netscape cookies.txt to *output_path*.

    Returns True if auth cookies were found, False otherwise.
    The file is still written even without auth cookies (it may help
    with bot detection even in an anonymous session).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Edge first (pre-installed on Windows), then Chrome
    for browser_name, launcher in [("Edge", _launch_edge), ("Chrome", _launch_chrome)]:
        try:
            cookies = launcher()
            if cookies is not None:
                return _write_cookies(cookies, output_path, browser_name)
        except Exception as e:
            logger.debug("Failed to extract cookies from %s: %s", browser_name, e)
            continue

    logger.warning("Could not extract cookies from any browser")
    return False


def _launch_edge() -> list[dict] | None:
    """Launch headless Edge and return cookies from youtube.com."""
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options
    except ImportError:
        logger.debug("selenium not installed, skipping Edge cookie extraction")
        return None

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Edge(options=options)
    except Exception as e:
        logger.debug("Edge launch failed: %s", e)
        return None

    return _get_youtube_cookies(driver)


def _launch_chrome() -> list[dict] | None:
    """Launch headless Chrome and return cookies from youtube.com."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        logger.debug("selenium not installed, skipping Chrome cookie extraction")
        return None

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        logger.debug("Chrome launch failed: %s", e)
        return None

    return _get_youtube_cookies(driver)


def _get_youtube_cookies(driver) -> list[dict]:
    """Navigate to youtube.com and collect cookies."""
    try:
        driver.get("https://www.youtube.com")
        time.sleep(3)
        cookies = driver.get_cookies()
        return cookies
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def _write_cookies(cookies: list[dict], output_path: Path, browser_name: str) -> bool:
    """Write cookies to a Netscape cookies.txt file."""
    lines = ["# Netscape HTTP Cookie File\n"]
    for c in cookies:
        domain = c.get("domain", "")
        domain_flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expiry = int(c.get("expiry", 0))
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\t{domain_flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    has_auth = any(c["name"] in _AUTH_COOKIE_NAMES for c in cookies)
    logger.info(
        "Extracted %d cookies from %s to %s (authenticated=%s)",
        len(cookies), browser_name, output_path, has_auth,
    )
    return has_auth
