
from typing import Any, List

from langchain_core.tools import StructuredTool, tool
from langchain_tavily import TavilySearch

from logger import logger
from index import get_retriever, index_pdf_folder
from settings import settings


index_pdf_folder(settings.PDF_DIR)  # 1. убеждаемся что всё проиндексировано            # 2. открываем БД для поиска
_local_retriever = get_retriever()
_online_retriever = TavilySearch(tavily_api_key=settings.TAVILY_API_KEY)


@tool
def web_search_tool(query: str) -> str:
    """
    Searches the internet for current information.
    Use this tool when: the question is not related to Orthodox Christianity,
    or when retriever_tool did not return relevant results.
    """
    docs: StructuredTool = _online_retriever.invoke(query)

    if not docs.get('results'):
        logger.error('Tavily не преслал релевантных документов')
        return 'Tavily не преслал релевантных документов'

    results = []
    if docs.get('answer'):
        results.append(f'Общий ответ на запрос "{query}" = {docs.get('answer')}')

    for numb, doc in enumerate(docs.get('results')):
        if doc.get('score') and doc.get('score') > 0.5:
            results.append(f'\n{numb} ответ на запрос = {doc.get('content', '')}')

    return "\n\n".join(results)


@tool
def retrivier_tool(query: str) -> str:
    """
    Searches and returns information from local Orthodox Christian documents database.
    Use this tool for questions about Orthodox theology, saints, spiritual practices,
    church history, and religious texts.
    """
    docs = _local_retriever.invoke(query)
    if not docs:
        logger.info('Retriever не вернул релевантных документов по запросу')
        return "I found no relevant information in the Orthodox Documents"

    results = []
    logger.info(f'Retriever вернул {len(docs)} документов')
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retrivier_tool, web_search_tool] # Here i should make an automatic construct for tools list!