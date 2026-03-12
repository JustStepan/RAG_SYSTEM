# 📚 RAG System — Local PDF Knowledge Base with AI Agent
*Автор: Степан Д. @StefanDKO*

> Локальная RAG-система для интеллектуального поиска по PDF-документам с агентом на LangGraph.  
> В перспективе — поддержка различных форматов: DOCX, TXT, EPUB, веб-страницы и структурированные данные.

---

## Что это

RAG System — это локальный AI-ассистент, который отвечает на вопросы по загруженным вами документам. Система векторизует PDF-файлы, сохраняет их в ChromaDB и использует LangGraph-агента для интеллектуального поиска и синтеза ответов.

Если документы не содержат нужной информации — агент автоматически переключается на веб-поиск через Tavily.

---

## Архитектура

```
PDF Folder
    │
    ▼
[Indexer]
  ├── PyPDFLoader        — загрузка и парсинг PDF
  ├── RecursiveTextSplitter — разбивка на чанки (512 токенов, overlap 100)
  └── OpenAIEmbeddings   — векторизация через локальную embedding-модель
    │
    ▼
[ChromaDB]              — персистентное векторное хранилище
    │
    ▼
[LangGraph Agent]
  ├── State: AgentState (messages history)
  ├── Node: call_llm    — вызов локальной LLM
  ├── Node: take_action — выполнение tool calls
  └── Conditional edge: should_continue → loop / END
    │
    ├── Tool: retriever_tool   — поиск по ChromaDB (top-5 по similarity)
    └── Tool: web_search_tool  — веб-поиск через Tavily (fallback)
    │
    ▼
[MemorySaver]           — сохранение контекста диалога между сообщениями
    │
    ▼
CLI Interface           — интерактивный диалог в терминале
```

---

## Поток обработки запроса

```
Пользователь задаёт вопрос
        │
        ▼
   LLM анализирует вопрос
        │
        ├── Нужен поиск? ──► retriever_tool (ChromaDB)
        │                          │
        │                    Нет результатов?
        │                          │
        │                          └──► web_search_tool (Tavily)
        │
        └── Достаточно данных? ──► Генерирует ответ со ссылками на источники
```

---

## Технологический стек

| Компонент | Технология |
|-----------|-----------|
| Агентный фреймворк | LangGraph |
| LLM (локальная) | LM Studio (Qwen и другие совместимые модели) |
| Embedding модель | OpenAI-compatible (локально через LM Studio) |
| Векторная БД | ChromaDB (персистентное хранилище) |
| Загрузка документов | LangChain PyPDFLoader |
| Веб-поиск (fallback) | Tavily Search API |
| Память диалога | LangGraph MemorySaver |
| Настройки | Pydantic Settings |
| Логирование | Loguru |

---

## Структура проекта

```
RAG_SYSTEM/
├── src/
│   ├── agent.py          ← AgentState, узлы графа (call_llm, take_action)
│   ├── node_builder.py   ← сборка StateGraph + MemorySaver
│   ├── index.py          ← индексация PDF, работа с ChromaDB
│   ├── tools.py          ← retriever_tool + web_search_tool (Tavily)
│   ├── models.py         ← инициализация LLM и embedding моделей
│   ├── prompts.py        ← системный промпт агента
│   ├── settings.py       ← конфигурация (Pydantic Settings)
│   ├── logger.py         ← Loguru логгер
│   ├── main.py           ← точка входа, CLI-диалог
│   └── storage/          ← ChromaDB данные
├── PDF/                  ← папка для ваших PDF документов
├── .env                  ← ключи и настройки
├── .env.example
└── pyproject.toml
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/JustStepan/RAG_SYSTEM.git
cd RAG_SYSTEM
uv sync
```

### 2. Настроить `.env`

```bash
cp .env.example .env
```

```env
COLLECTION_NAME=my_docs
TAVILY_API_KEY=ваш_ключ
LLM_MODEL=qwen3.5-9b           # название модели в LM Studio
EMBEDDING_MODEL=text-embedding-berta-uncased
LLM_URL=http://localhost:1234/v1
CHUNK_SIZE=512
CHUNK_OVERLAP=100
```

### 3. Добавить PDF документы

Положите ваши PDF-файлы в папку `src/PDF/`.

### 4. Запустить LM Studio

Откройте LM Studio, загрузите модель и нажмите **Start Server**.

### 5. Запустить агента

```bash
uv run src/main.py
```

При первом запуске система автоматически проиндексирует все PDF в папке. Новые документы добавляются инкрементально — уже проиндексированные пропускаются.

---

## Как работает индексация

- При запуске система проверяет, какие PDF уже есть в ChromaDB (по пути файла в метаданных)
- Новые файлы разбиваются на чанки по 512 токенов с перекрытием 100 токенов
- Пустые и невалидные чанки фильтруются
- Большие файлы (5000+ чанков) добавляются пакетами

---

## Особенности агента

**Два инструмента поиска:**
- `retriever_tool` — основной, ищет по локальной ChromaDB, возвращает top-5 релевантных чанков
- `web_search_tool` — fallback, вызывается когда локальная база не даёт результата; имеет порог релевантности (score ≥ 0.4)

**Память диалога:** через `MemorySaver` агент помнит контекст всего разговора в рамках одной сессии.

**Ограничение рекурсии:** `recursion_limit=10` защищает от бесконечных циклов tool-calling.

---

## Планы развития

- [ ] Поддержка DOCX, TXT, EPUB форматов
- [ ] Загрузка и индексация веб-страниц по URL
- [ ] Поддержка структурированных данных (CSV, JSON)
- [ ] Веб-интерфейс (FastAPI + React)
- [ ] Онлайн-режим через OpenRouter (Claude, GPT)
- [ ] Гибридный поиск (vector + BM25 keyword search)
- [ ] Оценка качества ответов (RAG evaluation)
- [ ] Docker-деплой

---

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов
- [LM Studio](https://lmstudio.ai/) — локальный сервер для LLM и embedding моделей
- Tavily API key — для веб-поиска ([бесплатный тариф](https://tavily.com))

---

## Автор

Степан — [@StefanDKO](https://t.me/StefanDKO) · [GitHub](https://github.com/JustStepan)
