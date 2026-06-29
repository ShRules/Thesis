import os
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
DATA_PATH = Path("data")
DB_PATH = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
def chunk_id(chunk) -> str:
    source = chunk.metadata.get("source", "")
    page = chunk.metadata.get("page", 0)
    content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{page}:{content_hash}"
def load_pdfs(data_path: Path):
    pdf_files = sorted(data_path.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_path.resolve()}")
    documents = []
    for pdf in pdf_files:
        try:
            docs = PyPDFLoader(str(pdf)).load()
            documents.extend(docs)
            print(f"Loaded {pdf.name} ({len(docs)} pages)")
        except Exception as e:
            print(f"Skipped {pdf.name}: {e}", file=sys.stderr)
    return documents
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    if not DATA_PATH.exists():
        sys.exit(f"Data directory not found: {DATA_PATH.resolve()}")
    print(f"Loading PDFs from {DATA_PATH}/ ...")
    documents = load_pdfs(DATA_PATH)
    print(f"Loaded {len(documents)} pages total.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    ids = [chunk_id(c) for c in chunks]
    print(f"Split into {len(chunks)} chunks.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    existing = set(db.get(include=[])["ids"])
    new_pairs = [(c, i) for c, i in zip(chunks, ids) if i not in existing]
    if not new_pairs:
        print("Nothing new to add. Vectorstore is up to date.")
        return
    print(
        f"Adding {len(new_pairs)} new chunks "
        f"(skipping {len(chunks) - len(new_pairs)} duplicates)..."
    )
    db.add_documents(
        documents=[c for c, _ in new_pairs],
        ids=[i for _, i in new_pairs],
    )
    print("Done. Documents processed and stored successfully.")
if __name__ == "__main__":
    main()