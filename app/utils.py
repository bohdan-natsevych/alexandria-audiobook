import os
import json
import time
import tempfile


def build_llm_extra_body(base_url, top_k=0, min_p=0, banned_tokens=None):
    """Build OpenAI-compatible extra_body sampling fields.

    CURSOR: Gemini's OpenAI-compat endpoint rejects top_k/min_p/banned_tokens
    even when zero; omit them for Google URLs. Other local servers may use them.
    """
    url = str(base_url or "").lower()
    if "generativelanguage.googleapis.com" in url or "/gemini" in url:
        return {}
    body = {}
    if top_k:
        body["top_k"] = top_k
    if min_p:
        body["min_p"] = min_p
    if banned_tokens:
        body["banned_tokens"] = banned_tokens
    return body


def atomic_json_write(data, target_path, max_retries=5):
    """Atomically write JSON data using a temp file and os.replace.

    Includes retry logic with exponential backoff for Windows file locking
    (Access is denied / file in use errors).
    """
    directory = os.path.dirname(target_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target_path)
                return
            except OSError as e:
                if attempt < max_retries - 1 and (
                    e.errno == 5
                    or "Access is denied" in str(e)
                    or "being used by another process" in str(e)
                ):
                    delay = 0.05 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
