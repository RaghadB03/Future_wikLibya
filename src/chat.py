# src/chat.py

from dotenv import load_dotenv
load_dotenv()

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding



PERSIST_DIR = "chroma_db"
COLLECTION = "libya_kb"

IDK = "The requested information is not available in the current knowledge sources."

#SAME models used during indexing
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")



def load_index() -> VectorStoreIndex:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )


def answer(
    index: VectorStoreIndex,
    question: str,
    top_k: int = 5,
    min_score: float = 0.35,
    debug: bool = True
):
    """
    Clean RAG pipeline:
    1) Retrieve top_k similar chunks
    2) Filter by similarity score
    3) Build context from strongest chunks
    4) Generate answer grounded only in context
    5) Return citations
    """

    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved = retriever.retrieve(question)
    
#     if not retrieved:
#         return IDK, []

#     # Filter by similarity threshold
#     strong = [
#         r for r in retrieved
#         if r.score is None or r.score >= min_score
#     ]

#     if not strong:
#         # fallback to top 2 if everything filtered
#         strong = retrieved[:2]

#     if debug:
#         scores = [r.score for r in strong if r.score is not None]
#         print(f"[debug] strong_scores={scores}")

#     # Build context from top 3 strong chunks
#     context_blocks = []
#     sources = []

#     for r in strong[:3]:
#         md = r.node.metadata or {}
#         context_blocks.append(r.node.get_text())
#         sources.append({
#             "score": r.score,
#             "title": md.get("title"),
#             "category": md.get("category"),
#             "url": md.get("source_url"),
#             "preview": r.node.get_text()[:220].replace("\n", " ")
#         })

#     context = "\n\n---\n\n".join(context_blocks)

#     prompt = f"""
# You are a Libya-focused assistant.
# Answer ONLY from the CONTEXT below.
# If the answer is not present in the CONTEXT, reply exactly:
# "{IDK}"

# CONTEXT:
# {context}

# QUESTION:
# {question}
# """.strip()

#     response = Settings.llm.complete(prompt).text.strip()
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(question)
    return response or IDK




def main():
    index = load_index()

    print("✅ Libya RAG Chatbot is ready.")
    print("Type a question. Type 'exit' to quit.")

    while True:
        q = input("\nYou: ").strip()

        if q.lower() in {"exit", "quit"}:
            break

        ans = answer(index, q)

        print("\nBot:", ans)
        print(ans.metadata)
        # if sources:
        #     print("\nCitations:")
        #     for i, s in enumerate(sources, 1):
        #         sc = s["score"]
        #         sc_str = f"{sc:.3f}" if isinstance(sc, (int, float)) else str(sc)
        #         print(f"  {i}) score={sc_str} | {s['category']} | {s['title']}")
        #         print(f"     {s['url']}")
        #         print(f"     {s['preview']}...")


if __name__ == "__main__":
    main()