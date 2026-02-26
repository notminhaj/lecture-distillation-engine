"""
Export YouTube cookies from Chrome via Selenium using a copy of the user's profile.

This avoids the DPAPI/app-bound encryption issue entirely — Selenium gets
cookies through Chrome's DevTools protocol, which returns them decrypted.

Usage:
    python scripts/selenium_cookies.py [output_path]
"""

import os
import shutil
import sys
import tempfile
import time


def main(output_path: str = "cache/cookies.txt") -> int:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    # Copy the Chrome profile to a temp dir so we don't conflict with a
    # running browser instance (Chrome locks its profile directory).
    src_profile = os.path.join(
        os.path.expanduser("~"),
        "AppData", "Local", "Google", "Chrome", "User Data",
    )
    if not os.path.isdir(src_profile):
        print(f"Chrome profile not found at {src_profile}")
        return 1

    tmp_profile = tempfile.mkdtemp(prefix="chrome_cookies_")
    print(f"Copying Chrome profile to {tmp_profile} ...")

    # Only copy the files we need (cookies, Local State) — full profile is huge
    for item in ("Local State",):
        src = os.path.join(src_profile, item)
        dst = os.path.join(tmp_profile, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Copy the Default profile directory (minimal set of files)
    src_default = os.path.join(src_profile, "Default")
    dst_default = os.path.join(tmp_profile, "Default")
    os.makedirs(dst_default, exist_ok=True)
    for item in ("Preferences", "Secure Preferences", "Login Data", "Cookies",
                 "Network"):
        src = os.path.join(src_default, item)
        dst = os.path.join(dst_default, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-data-dir={tmp_profile}")
    options.add_argument("--profile-directory=Default")

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Chrome launch failed: {e}")
        print("Trying Edge instead...")
        from selenium.webdriver.edge.options import Options as EdgeOptions
        options = EdgeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        # Edge uses a different profile path
        src_edge = os.path.join(
            os.path.expanduser("~"),
            "AppData", "Local", "Microsoft", "Edge", "User Data",
        )
        tmp_edge = tempfile.mkdtemp(prefix="edge_cookies_")
        for item in ("Local State",):
            src = os.path.join(src_edge, item)
            dst = os.path.join(tmp_edge, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        src_def = os.path.join(src_edge, "Default")
        dst_def = os.path.join(tmp_edge, "Default")
        os.makedirs(dst_def, exist_ok=True)
        for item in ("Preferences", "Secure Preferences", "Network"):
            src = os.path.join(src_def, item)
            dst = os.path.join(dst_def, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
        options.add_argument(f"--user-data-dir={tmp_edge}")
        options.add_argument("--profile-directory=Default")
        driver = webdriver.Edge(options=options)

    print("Browser started, navigating to YouTube...")
    driver.get("https://www.youtube.com")
    time.sleep(5)

    cookies = driver.get_cookies()
    print(f"Got {len(cookies)} cookies from browser")

    # Check for auth cookies
    auth_names = {"SID", "HSID", "SSID", "APISID", "SAPISID",
                  "LOGIN_INFO", "__Secure-1PSID", "__Secure-3PSID"}
    found_auth = [c["name"] for c in cookies if c["name"] in auth_names]
    print(f"Auth cookies found: {found_auth}")

    # Write Netscape cookies.txt
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {len(cookies)} cookies to {output_path}")

    driver.quit()

    # Cleanup temp dirs
    try:
        shutil.rmtree(tmp_profile, ignore_errors=True)
    except Exception:
        pass

    if not found_auth:
        print("\nWARNING: No auth cookies found. You may not be logged into YouTube in Chrome/Edge.")
        print("Log into YouTube in your browser and re-run this script.")
        return 1

    return 0


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "cache/cookies.txt"
    sys.exit(main(output))
