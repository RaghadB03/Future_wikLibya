from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
from pprint import pprint
from tavily import TavilyClient
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context


llm = OpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
Settings.llm = llm


def tavily_search(query: str):
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client.search(query=query, search_depth="basic", max_results=5)


search_tool = FunctionTool.from_defaults(
    fn=tavily_search,
    name="web_search",
    description="Search the web for recent or unknown information."
)


async def ask_agent():

    agent = ReActAgent(
        tools=[search_tool],
        llm=llm,
    )

    ctx = Context(agent)

    handler = await agent.run(
        "سعر الدينار الليبي مقابل الدولار في السوق الموازي اليوم في ليبيا؟",
        ctx=ctx
    )

    return handler.response

