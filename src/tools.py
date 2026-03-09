
from langchain_core.tools import tool

from logger import logger
from index import get_retriever, index_pdf_folder
from settings import settings


index_pdf_folder(settings.PDF_DIR)  # 1. убеждаемся что всё проиндексировано            # 2. открываем БД для поиска
_retriever = get_retriever()


@tool
def retrivier_tool(query: str) -> str:
    """
    This tool searches and returns the information from Orthodox Document
    """
    docs = _retriever.invoke(query)
    if not docs:
        logger.info('Retriever не вернул релевантных документов по запросу')
        return "I found no relevant information in the Orthodox Documents"

    results = []
    logger.info(f'Retriever вернул {len(docs)} документов')
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retrivier_tool] # Here i should make an automatic construct of tools!