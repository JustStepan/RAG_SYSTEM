from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from settings import settings


local_model = ChatOpenAI(
    model=settings.LLM_MODEL,
    base_url=settings.LLM_URL,
    api_key="lm-studio-dummy-key",
    temperature=0,
)

local_embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    base_url=settings.LLM_URL,
    api_key="lm-studio-dummy-key",
    check_embedding_ctx_length=False,  # Отключаем проверку длины контекста
    request_timeout=120,
)

# === ONLINE MODELS ===
# Нужно доделать

# online_model = ChatOpenAI(
#     model="?",
#     base_url="?",
#     api_key=settings.OPENAI_API_KEY,
#     temperature=0,
# )

# online_embeddings = OpenAIEmbeddings(
#     model="?",
#     base_url="?",
#     api_key=settings.OPENAI_API_KEY,
#     check_embedding_ctx_length=False,  # Отключаем проверку длины контекста
#     request_timeout=60,
# )
