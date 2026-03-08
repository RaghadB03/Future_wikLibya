from dotenv import load_dotenv
load_dotenv()

import os
import asyncio

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from tavily import TavilyClient
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

from RAG import ask_rag

llm = OpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

Settings.llm = llm


def tavily_search(query: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    result = client.search(query=query, search_depth="basic", max_results=5)
    
    results = result.get("results", [])
    if not results:
        return f"No web results found for: {query}"

    output = [f"Search query: {query}"]

    for i, item in enumerate(results, start=1):
        title = item.get("title", "No title")
        content = item.get("content", "No content")
        url = item.get("url", "No URL")

        output.append(
            f"Result {i}:\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"URL: {url}"
        )

    return "\n\n".join(output)


search_tool = FunctionTool.from_defaults(fn=tavily_search, name="web_search", description="Search external online sources.")

rag_tool = FunctionTool.from_defaults(fn=ask_rag, name="rag_search", description="Search the internal source.")


agent = ReActAgent(
    tools=[search_tool, rag_tool],
    llm=llm,
    verbose=True,
    system_prompt="""
You are a helpful AI assistant specialized in answering questions about Libya.

Rules:
1. Only answer questions relevant to Libya.
2. You MUST use a tool before answering.
3. Use rag_search when the answer is likely to be found in the internal knowledge source.
4. Use web_search when the answer is likely to require external online sources.
5. If one tool does not return useful information, you may use the other tool before answering.
6. Keep the final answer short and direct.
"""
)

ctx = Context(agent)




async def main():
    question ="كم سعر الدينار الليبي مقابل الدولار الأمريكي في السوق السوداء اليوم؟"

    handler = await agent.run(question, ctx=ctx)

    for call in handler.tool_calls:
        print(f"Tool: {call.tool_name}")
        print(f"Input: {call.tool_kwargs}")
        print()

    print(handler.response.blocks[0].text)


if __name__ == "__main__":
    asyncio.run(main())