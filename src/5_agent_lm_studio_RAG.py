import os
from typing import Annotated, Sequence, TypedDict
from pathlib import Path

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END, START
from logger import logger


# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve()
db_storage = BASE_DIR.parent / "storage"
pdf_path = BASE_DIR.parent / "PDF/Kon_Vvedenie.pdf"
collection_name = "orthodox_texts"


# ===== MODELS =====
model = ChatOpenAI(
    model="qwen3.5-9b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio-dummy-key",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-berta-uncased",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio-dummy-key",
    check_embedding_ctx_length=False,  # Отключаем проверку длины контекста
    request_timeout=60,
)


# ===== INDEXING =====
def pdf_loader(pdf_path: Path) -> PyPDFLoader:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'PDF file not found here: "{pdf_path}"')

    pdf_loader = PyPDFLoader(pdf_path)

    try:
        pages = pdf_loader.load()
        logger.info(f"PDF документ был загружет и имеет {len(pages)} страниц")
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise

    return pages


def get_split_pages(pdf_path: Path) -> RecursiveCharacterTextSplitter:
    pages = pdf_loader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100
    )
    docs = text_splitter.split_documents(pages)
    logger.info(f'Документ разбит на {len(docs)} частей')
    
    return docs

def create_embedings_db(pdf_path: Path):
    logger.info('Обращение к БД')
    try:
        if os.path.exists(db_storage):
            existing_db = Chroma(
                persist_directory=db_storage,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
            ln = existing_db._collection.count()
            if ln > 0:
                logger.info(f'База данных уже существует ее размер: {ln}')
                logger.info(f'Используем эмбедер: {embeddings.model}')
                return existing_db

        # If Db is empty se below we create documents and create new DB
        logger.info(f'Создаем новую БД...')
        os.makedirs(db_storage, exist_ok=True)
        documents = get_split_pages(pdf_path)
        # убираем пустые блоки
        valid_docs = [
            doc
            for doc in documents
            if doc.page_content
            and isinstance(doc.page_content, str)
            and doc.page_content.strip()
        ]

        logger.info(
            f"Filtered: {len(documents)} -> {len(valid_docs)} valid chunks"
        )

        vectorstore = Chroma.from_documents(
            documents=valid_docs,
            embedding=embeddings,
            persist_directory=db_storage,
            collection_name=collection_name,
        )
        logger.info(f"База данных ChromaDB успешно создана")
        logger.info(f'Использован эмбедер: {embeddings.model}')

    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {str(e)}")
        raise

    return vectorstore


def get_retriever():
    return create_embedings_db(pdf_path).as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )


# ===== TOOLS =====
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


# ===== AGENT =====
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> str:
    """Check if the last message contains tool_calls"""

    result = state["messages"][-1]
    logger.info(f'Применена логическая нода для сообщения: {state["messages"][-1].content[:70]}')
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


tools = [retrivier_tool]
if tools:
    logger.info(f'Список используемых тулов: {tools}')
llm = model.bind_tools(tools)
if llm:    
    logger.info(f'Использована модель: {llm.model}')
system_prompt = """
You are an intelligent AI assistant who answers questions about Orthodox document based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the orthodox data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
Always in russian language or in the language of user question. 
"""
tools_dict = {
    our_tool.name: our_tool for our_tool in tools
}  # Creating a dictionary of our tools
print(tools_dict)

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        logger.info(
            f"Вызванный тул: {t['name']} с запросом: {t['args'].get('query', 'No query provided')}"
        )

        if not t["name"] in tools_dict:  # Checks if a valid tool is present
            logger.error(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            logger.info(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(
                tool_call_id=t["id"], name=t["name"], content=str(result)
            )
        )

    logger.info("Выполнение тула завершено! Двигаемся обратно к модели!")
    return {"messages": results}


# ===== BUILD NODES =====
def builder() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)

    graph.add_edge(START, "llm")
    graph.add_edge("retriever_agent", "llm")

    graph.add_conditional_edges(
        source="llm",
        path=should_continue,
        path_map={True: "retriever_agent", False: END},
    )
    return graph.compile()


# ===== ENTRYPOINT =====
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
