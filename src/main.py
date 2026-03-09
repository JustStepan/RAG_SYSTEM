from langchain_core.messages import HumanMessage

from logger import logger
from node_builder import builder


rag_agent = builder()
if rag_agent:
    logger.info('Агент инициирован, граф успешно собран.')


def running_agent():
    print("\n=== RAG AGENT===")

    while True:
        user_input = input("\nКакой у вас вопрос сударь: ")
        if user_input.lower() in ["exit", "quit", "выход"]:
            break

        messages = [
            HumanMessage(content=user_input)
        ]  # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


running_agent()


# index_pdf_folder(settings.PDF_FOLDER)  # 1. убеждаемся что всё проиндексировано
# retriever = get_retriever()             # 2. открываем БД для поиска