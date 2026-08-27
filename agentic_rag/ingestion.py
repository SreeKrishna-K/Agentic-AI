from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from utils.logging import log_panel, log_rule, log_table

load_dotenv()

PERSIST_DIRECTORY = "./.chroma"
COLLECTION_NAME = "rag-chroma"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 0

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def _load_documents():
    rows: list[tuple[str, str, str]] = []
    docs_list = []
    for url in URLS:
        try:
            loaded = UnstructuredLoader(
                web_url=url,
                chunking_strategy="basic",
                max_characters=1_000_000,
            ).load()
            docs_list.extend(loaded)
            rows.append((url, str(len(loaded)), "[green]loaded[/green]"))
        except Exception as exc:
            rows.append((url, "0", f"[red]failed[/red]  {exc}"))

    log_table(
        "Source pages",
        ["URL", "Docs", "Status"],
        rows,
        justify={"Docs": "right"},
    )

    if not docs_list:
        raise RuntimeError("No documents were loaded from any source URL.")

    return docs_list


def _ingest(vectorstore: Chroma) -> int:
    log_rule("INGEST  ·  load source documents", "bold cyan")
    docs_list = _load_documents()
    log_panel(
        "Loaded documents",
        f"{len(docs_list)} document(s) from {len(URLS)} URL(s)",
        "cyan",
    )

    log_rule("INGEST  ·  split into chunks", "bold cyan")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    log_panel(
        "Chunks",
        f"{len(doc_splits)} chunk(s)  ·  size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}",
        "cyan",
    )

    log_rule("INGEST  ·  embed and persist", "bold green")
    for doc in doc_splits:
        doc.metadata["embedding_model"] = EMBEDDING_MODEL
    vectorstore.add_documents(doc_splits)
    log_panel(
        "Vector store",
        f"Wrote {len(doc_splits)} chunk(s) with {EMBEDDING_MODEL} to '{COLLECTION_NAME}' at {PERSIST_DIRECTORY}",
        "green",
    )
    return len(doc_splits)


def _make_vectorstore(embeddings: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def _stored_embedding_model(vectorstore: Chroma) -> str | None:
    snapshot = vectorstore.get(include=["metadatas"], limit=1)
    if not snapshot["ids"]:
        return None
    metadata = (snapshot["metadatas"] or [{}])[0] or {}
    return metadata.get("embedding_model")


def get_retriever():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = _make_vectorstore(embeddings)
    stored_model = _stored_embedding_model(vectorstore)
    stored_count = len(vectorstore.get(include=[])["ids"])

    if stored_count and stored_model != EMBEDDING_MODEL:
        log_rule("INGEST  ·  rebuild (embedding model changed)", "bold magenta")
        log_panel(
            "Vector store",
            f"Replacing {stored_count} embedding(s) "
            f"({stored_model or 'unknown/legacy'} → {EMBEDDING_MODEL})",
            "magenta",
        )
        vectorstore.delete_collection()
        vectorstore = _make_vectorstore(embeddings)
        stored_count = 0

    if not stored_count:
        _ingest(vectorstore)
    else:
        log_rule("INGEST  ·  skip (already populated)", "bold yellow")
        log_panel(
            "Vector store",
            f"Collection '{COLLECTION_NAME}' already has {stored_count} "
            f"{EMBEDDING_MODEL} embedding(s) at {PERSIST_DIRECTORY}",
            "yellow",
        )

    return vectorstore.as_retriever()


retriever = get_retriever()
