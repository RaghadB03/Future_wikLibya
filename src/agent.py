from dotenv import load_dotenv
load_dotenv()

import os
from pinecone import Pinecone

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def get_index() -> VectorStoreIndex:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("wiklibya")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

from tavily import TavilyClient
def tavily_search(query:str):
    'Function that search the web given a query, returns found resources and their metadata.'
    client = TavilyClient(os.getenv('TAVILY_API_KEY'))
    response = client.search(
        query=query,
        search_depth="basic"
    )
    return response

result = tavily_search('من هو محافظ مركز ليبيا المركزي الحالي؟')
print(result)


from llama_index.core.tools import FunctionTool
search_function = FunctionTool.from_defaults(tavily_search)

response = llm.predict_and_call(tools=[search_function],
                                user_msg='من هو محافظ مصرف ليبيا المركزي الحالي؟',
                                verbose= True
                                )

pprint(response.response)


from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

agent = ReActAgent(
    tools=[search_function, currency_function],
    llm=llm,
    verbose=True,
)

ctx = Context(agent)


handler = await agent.run('من هو المحافظ الحالي لمصرف ليبيا المركزي؟', ctx=ctx)

print(handler.response)

print(handler.tool_calls)