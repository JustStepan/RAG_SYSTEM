import os
import tqdm
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import settings
from logger import logger
from models import local_embeddings


def get_or_create_db():
    os.makedirs(settings.DB_STORAGE, exist_ok=True)
    # Chroma сама решает — открыть существующую или создать пустую
    db = Chroma(
        persist_directory=settings.DB_STORAGE,
        embedding_function=local_embeddings,
        collection_name=settings.COLLECTION_NAME,
    )
    logger.info(f'БД открыта. Документов в коллекции: {db._collection.count()}')
    return db


def get_indexed_sources(db) -> set[Path]:
    all_metadatas = []
    offset = 0
    while True:
        batch = db._collection.get(limit=1000, offset=offset, include=["metadatas"])
        if batch.get('metadatas'):
            all_metadatas.extend(batch['metadatas'])
            offset += 1000
        else:
            break
    return set(Path(m['source']) for m in all_metadatas if m)


def index_pdf_folder(pdf_folder):
    if not pdf_folder.exists() or not pdf_folder.is_dir():
        raise NotADirectoryError(f"Путь {pdf_folder} не является корректной папкой")

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        logger.info(f"В папке {pdf_folder} не найдено PDF файлов.")
        return
    
    logger.info(f'Папка содержит {len(pdf_files)} PDF файлов.')

    chroma_db = get_or_create_db()
    existing_docs: set[Path] = get_indexed_sources(chroma_db)
    
    valid_pdf_files = [f for f in pdf_files if f not in existing_docs]
    logger.info(f'Количество новых документов: {len(valid_pdf_files)}')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    for pdf_path in tqdm.tqdm(valid_pdf_files, desc="Индексация PDF", unit="файл"):
        logger.info(f'Добавляем файл "{pdf_path.name}" в базу данных.')
        docs = get_split_pages(pdf_path, text_splitter)

        valid_docs = [
            doc
            for doc in docs
            if doc.page_content
            and isinstance(doc.page_content, str)
            and doc.page_content.strip()
        ]

        if not valid_docs:
            logger.warning(f'Нет валидных чанков для "{pdf_path.name}", пропускаем.')
            continue
        
        logger.info(
            f"Файл '{pdf_path.name}' отфильтрован: {len(docs)} -> {len(valid_docs)} валидных чанков"
        )

        if len(valid_docs) >= 5000:
            logger.info(f'Разделяем чанки: "{pdf_path.name}"')

        while len(valid_docs) >= 5000:
            chroma_db.add_documents(valid_docs[:5000])
            valid_docs = valid_docs[5000:]
        chroma_db.add_documents(valid_docs)
    
        logger.info(f'Проиндексирован файл: "{pdf_path.name}"')


def get_split_pages(pdf_path: Path, splitter: RecursiveCharacterTextSplitter):
    pdf_loader = PyPDFLoader(pdf_path)

    try:
        pages = pdf_loader.load()
        logger.info(f"PDF документ '{pdf_path.name}' был загружет и имеет {len(pages)} страниц")
    except Exception as e:
        logger.error(f"Ошибка загрузки документа '{pdf_path.name}': {e}")
        return []

    docs = splitter.split_documents(pages)
    logger.info(f'Документ "{pdf_path.name}" разбит на {len(docs)} частей')

    return docs


def get_retriever():
    return get_or_create_db().as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
