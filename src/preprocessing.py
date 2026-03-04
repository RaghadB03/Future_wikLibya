import os
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding


REPORT_PATH = "reports/source_validation_report.json"
URLS_FILE = "data/urls.json"


INDEX_NAME = "wiklibya"   



Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def extract_text_from_html(html: str) -> str:
    """Convert HTML to clean text for indexing (remove navigation/UI noise)."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])


def fetch(url: str, timeout: int, user_agent: str) -> Tuple[Optional[requests.Response], Optional[str]]:
    """Fetch a URL safely (returns (response, error))."""
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": user_agent},
            allow_redirects=True,
        )
        return r, None
    except Exception as e:
        return None, str(e)


def load_registry() -> Dict[str, Any]:
    """Load enterprise source registry (data/urls.json)."""
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ok_ids_from_report() -> List[str]:
    """Read validation report and keep only sources with status='ok'."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return [r["id"] for r in report.get("results", []) if r.get("status") == "ok"]


def main():
    reg = load_registry()
    ok_ids = set(load_ok_ids_from_report())

    defaults = reg.get("defaults", {}).get("request", {})
    timeout = int(defaults.get("timeout_seconds", 20))
    user_agent = str(defaults.get("user_agent", "Mozilla/5.0 (compatible; LibyaRAG/1.0)"))

    sources = reg.get("sources", [])
    ok_sources = [s for s in sources if s.get("id") in ok_ids]

    if not ok_sources:
        raise RuntimeError("No OK sources found. Run validate_sources.py first.")

    print(f"✅ OK sources to index: {len(ok_sources)} / {len(sources)}")

    documents: List[Document] = []
    skipped: List[Dict[str, Any]] = []

    for src in ok_sources:
        url = src["url"]
        print(f"🌍 Fetching OK source: {src['title']} | {url}")

        r, err = fetch(url, timeout=timeout, user_agent=user_agent)
        if err or r is None:
            skipped.append({"id": src["id"], "url": url, "reason": f"Fetch error: {err}"})
            continue

        if r.status_code >= 400:
            skipped.append({"id": src["id"], "url": url, "reason": f"HTTP {r.status_code}"})
            continue

        text = extract_text_from_html(r.text)
        if len(text) < 500:
            skipped.append({"id": src["id"], "url": url, "reason": f"Low text extracted: {len(text)}"})
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "source_id": src["id"],
                    "category": src["category"],
                    "title": src["title"],
                    "source_url": str(r.url), 
                    "description": src["description"],
                    "content_type": r.headers.get("Content-Type"),
                },
            )
        )

    if not documents:
        raise RuntimeError("No documents fetched for indexing. Something is wrong with OK sources.")

    print(f"\n✅ Documents ready: {len(documents)}")
    if skipped:
        print(f"⚠️ Skipped during build: {len(skipped)}")
        for s in skipped[:10]:
            print(" -", s)

  #chunking
    splitter = SentenceSplitter(
        chunk_size=1024,     
        chunk_overlap=200
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✅ Nodes created: {len(nodes)}")

    # pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(INDEX_NAME)

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Insert nodes into Pinecone
    index.insert_nodes(nodes)
    print("\n✅ Pinecone index updated successfully!")
    print(f"   Index: {INDEX_NAME}")


if __name__ == "__main__":
    main()