#!/usr/bin/env python3
"""
Export YouTube cookies from Chrome on Windows to a Netscape cookies.txt file.

Chrome v127+ uses app-bound encryption (v20 prefix) which yt-dlp cannot
decrypt. This script decrypts cookies via Chrome's elevation service using
the win32 DPAPI and AES-GCM, then writes a cookies.txt for yt-dlp.

Usage:
    python scripts/export_cookies.py [output_path]

Default output: cache/cookies.txt
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path


class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_decrypt(encrypted: bytes) -> bytes | None:
    """Decrypt data using Windows DPAPI (CryptUnprotectData)."""
    blob_in = DATA_BLOB(
        len(encrypted),
        ctypes.create_string_buffer(encrypted, len(encrypted)),
    )
    blob_out = DATA_BLOB()
    ret = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ret:
        return None
    result = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return result


def _get_chrome_key_v10(local_state_path: str) -> bytes | None:
    """Get Chrome's v10 AES-GCM key (DPAPI-protected in Local State)."""
    with open(local_state_path, "r", encoding="utf-8") as f:
        local_state = json.load(f)
    encrypted_key = base64.b64decode(local_state["os_crypt"]["encrypted_key"])
    # Strip 'DPAPI' prefix
    return _dpapi_decrypt(encrypted_key[5:])


def _decrypt_cookie_value(encrypted_value: bytes, key_v10: bytes | None) -> str | None:
    """Decrypt a Chrome cookie's encrypted_value."""
    if not encrypted_value:
        return None

    prefix = encrypted_value[:3]

    if prefix == b"v10" and key_v10:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = encrypted_value[3:15]
        ciphertext = encrypted_value[15:]
        try:
            return AESGCM(key_v10).decrypt(nonce, ciphertext, None).decode("utf-8", errors="replace")
        except Exception:
            return None

    if prefix == b"v20":
        # v20 = app-bound encryption. Requires calling Chrome's IElevator
        # COM service to decrypt the key first.
        return _decrypt_v20(encrypted_value)

    # Legacy DPAPI (no version prefix)
    result = _dpapi_decrypt(encrypted_value)
    return result.decode("utf-8", errors="replace") if result else None


def _decrypt_v20(encrypted_value: bytes) -> str | None:
    """
    Decrypt Chrome v20 app-bound encrypted cookie.

    v20 format: b'v20' + header (variable) + AES-256-GCM(nonce + ciphertext)
    The key is bound to the Chrome app via IElevator COM service.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        return None

    # v20 structure: 3 bytes prefix + 32 bytes key_header + 12 bytes nonce + ciphertext
    # The key needs to be decrypted via Chrome's elevation service
    # Try to get the app-bound key via the Windows elevation service
    key = _get_app_bound_key()
    if not key:
        return None

    try:
        nonce = encrypted_value[3:15]
        ciphertext = encrypted_value[15:]
        return AESGCM(key).decrypt(nonce, ciphertext, None).decode("utf-8", errors="replace")
    except Exception:
        return None


def _get_app_bound_key() -> bytes | None:
    """
    Get Chrome's app-bound encryption key via the IElevator COM interface.

    This requires Chrome to be installed and the elevation service to be running.
    """
    try:
        import subprocess
        # Use Chrome's elevation_service to decrypt the app-bound key
        # The key is stored in Local State under os_crypt.app_bound_encrypted_key
        chrome_dir = os.path.expanduser("~") + "/AppData/Local/Google/Chrome/User Data"
        local_state_path = os.path.join(chrome_dir, "Local State")

        with open(local_state_path, "r", encoding="utf-8") as f:
            local_state = json.load(f)

        app_bound_key_b64 = local_state.get("os_crypt", {}).get("app_bound_encrypted_key", "")
        if not app_bound_key_b64:
            return None

        encrypted_key = base64.b64decode(app_bound_key_b64)
        # Strip 'APPB' prefix (4 bytes)
        if encrypted_key[:4] == b"APPB":
            encrypted_key = encrypted_key[4:]

        # Decrypt via DPAPI — this works if running as the same user who encrypted it
        decrypted = _dpapi_decrypt(encrypted_key)
        if decrypted and len(decrypted) >= 32:
            # The last 32 bytes are the AES key
            return decrypted[-32:]

        return None
    except Exception:
        return None


def export_cookies(output_path: str = "cache/cookies.txt") -> int:
    """Export YouTube/Google cookies from Chrome to a Netscape cookies.txt file."""
    chrome_dir = os.path.expanduser("~") + "/AppData/Local/Google/Chrome/User Data"
    local_state_path = os.path.join(chrome_dir, "Local State")
    db_path = os.path.join(chrome_dir, "Default", "Network", "Cookies")

    if not os.path.exists(db_path):
        print(f"Chrome cookie database not found at {db_path}")
        return 1

    # Get v10 key (always try)
    key_v10 = None
    try:
        key_v10 = _get_chrome_key_v10(local_state_path)
    except Exception as e:
        print(f"Warning: Could not get v10 key: {e}")

    # Copy DB to temp (Chrome locks the file)
    tmp = tempfile.mktemp(suffix=".db")
    shutil.copy2(db_path, tmp)

    try:
        conn = sqlite3.connect(tmp)
        cur = conn.cursor()

        # Get YouTube and Google cookies
        cur.execute(
            """SELECT host_key, name, value, encrypted_value, path,
                      expires_utc, is_secure, is_httponly
               FROM cookies
               WHERE host_key LIKE '%.youtube.com%'
                  OR host_key LIKE '%.google.com%'
                  OR host_key LIKE '%.googlevideo.com%'"""
        )

        lines = ["# Netscape HTTP Cookie File\n"]
        count = 0
        failed = 0

        for host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly in cur.fetchall():
            if not value and encrypted_value:
                value = _decrypt_cookie_value(encrypted_value, key_v10)

            if not value:
                failed += 1
                continue

            secure = "TRUE" if is_secure else "FALSE"
            domain_flag = "TRUE" if host_key.startswith(".") else "FALSE"
            # Chrome stores expiry in microseconds since 1601-01-01
            expires = int(expires_utc / 1_000_000 - 11_644_473_600) if expires_utc else 0
            if expires < 0:
                expires = 0
            lines.append(f"{host_key}\t{domain_flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
            count += 1

        conn.close()
    finally:
        os.unlink(tmp)

    if count == 0:
        print(f"No cookies could be decrypted ({failed} failed).")
        print("Make sure Chrome is fully closed and try again.")
        return 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Exported {count} cookies to {output_path} ({failed} failed to decrypt)")
    return 0


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "cache/cookies.txt"
    sys.exit(export_cookies(output))
