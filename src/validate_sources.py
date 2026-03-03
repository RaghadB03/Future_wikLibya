import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


URLS_FILE = "data/urls.json"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

REPORT_PATH = REPORTS_DIR / "source_validation_report.json"


def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join([ln for ln in lines if ln])
    return text


def fetch_with_retries(
    url: str,
    timeout: int,
    max_retries: int,
    user_agent: str
) -> Tuple[Optional[requests.Response], Optional[str]]:
    last_err = None
    headers = {"User-Agent": user_agent}

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            return r, None
        except Exception as e:
            last_err = str(e)
            # small backoff
            time.sleep(0.75 * (attempt + 1))

    return None, last_err


def load_registry() -> Dict[str, Any]:
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError("urls.json must be an object with metadata/defaults/sources.")
    if "sources" not in data or not isinstance(data["sources"], list):
        raise RuntimeError("urls.json must contain a 'sources' list.")
    return data


def main():
    reg = load_registry()
    defaults = reg.get("defaults", {}).get("request", {})

    timeout = int(defaults.get("timeout_seconds", 20))
    max_retries = int(defaults.get("max_retries", 2))
    min_text_chars = int(defaults.get("min_text_chars", 800))
    user_agent = str(defaults.get("user_agent", "Mozilla/5.0 (compatible; LibyaRAG/1.0)"))

    results: List[Dict[str, Any]] = []
    ok_count = 0

    for src in reg["sources"]:
        src_id = src.get("id")
        url = src.get("url")
        title = src.get("title")
        category = src.get("category")

        item_result: Dict[str, Any] = {
            "id": src_id,
            "title": title,
            "category": category,
            "url": url,
            "status": "unknown",
            "http_status": None,
            "final_url": None,
            "content_type": None,
            "text_chars": 0,
            "error": None
        }

        # Basic schema checks
        missing = [k for k in ("id", "category", "title", "url", "description") if not src.get(k)]
        if missing:
            item_result["status"] = "fail"
            item_result["error"] = f"Missing fields: {missing}"
            results.append(item_result)
            continue

        if not is_valid_url(url):
            item_result["status"] = "fail"
            item_result["error"] = "Invalid URL format (must include http/https)."
            results.append(item_result)
            continue

        # Fetch
        r, err = fetch_with_retries(url, timeout=timeout, max_retries=max_retries, user_agent=user_agent)
        if err or r is None:
            item_result["status"] = "fail"
            item_result["error"] = f"Request failed: {err}"
            results.append(item_result)
            continue

        item_result["http_status"] = r.status_code
        item_result["final_url"] = str(r.url)
        item_result["content_type"] = r.headers.get("Content-Type")

        # Status code gate
        if r.status_code >= 400:
            item_result["status"] = "fail"
            item_result["error"] = f"HTTP error status: {r.status_code}"
            results.append(item_result)
            continue

        # Parse text
        text = extract_text_from_html(r.text)
        item_result["text_chars"] = len(text)

        if len(text) < min_text_chars:
            item_result["status"] = "warn"
            item_result["error"] = f"Low extracted text (<{min_text_chars} chars). Might be dynamic/blocked."
            results.append(item_result)
            continue

        item_result["status"] = "ok"
        ok_count += 1
        results.append(item_result)

    report = {
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "warn": sum(1 for r in results if r["status"] == "warn"),
            "fail": sum(1 for r in results if r["status"] == "fail"),
        },
        "results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("✅ Validation complete.")
    print(f"Report saved to: {REPORT_PATH}")
    print("Summary:", report["summary"])


if __name__ == "__main__":
    main()