
"""Extract PDF text and build a FAISS vector store."""

import json
import logging
from pathlib import Path

import faiss
import pymupdf as fitz
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "vector_store"

INDEX_PATH = STORE_DIR / "medical.index"
CHUNKS_PATH = STORE_DIR / "chunks.json"
CONFIG_PATH = STORE_DIR / "config.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""
    text = " ".join(text.split())

    if not text:
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def load_pdf_chunks() -> list[dict]:
    """Extract text from all PDFs and preserve source/page metadata."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}"
        )

    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}. "
            "Place a PDF inside the data folder."
        )

    records = []

    for pdf_path in pdf_paths:
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_text = page.get_text("text").strip()

                    if not page_text:
                        logger.warning(
                            "No selectable text in %s, page %s. "
                            "A scanned PDF may require OCR.",
                            pdf_path.name,
                            page_number,
                        )
                        continue

                    for chunk_number, chunk in enumerate(
                        split_text(page_text), start=1
                    ):
                        records.append({
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk": chunk_number,
                        })

        except Exception as exc:
            logger.exception(
                "Could not read PDF %s: %s",
                pdf_path.name,
                exc,
            )

    if not records:
        raise ValueError(
            "No readable text was extracted. Check the PDF or use OCR "
            "if it contains scanned pages."
        )

    return records


def build_vector_store() -> None:
    """Embed extracted chunks and save a FAISS similarity index."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    records = load_pdf_chunks()

    logger.info("Extracted %d text chunks.", len(records))
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    model = SentenceTransformer(EMBEDDING_MODEL)

    vectors = model.encode(
        [record["text"] for record in records],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype="float32")

    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError("Unexpected embedding array shape.")

    # Normalized vectors + inner product = cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))

    CHUNKS_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    CONFIG_PATH.write_text(
        json.dumps({
            "embedding_model": EMBEDDING_MODEL,
            "dimension": int(vectors.shape[1]),
            "count": len(records),
            "metric": "cosine_similarity_via_normalized_inner_product",
        }, indent=2),
        encoding="utf-8",
    )

    logger.info("FAISS index saved: %s", INDEX_PATH)
    logger.info("Chunk metadata saved: %s", CHUNKS_PATH)
    logger.info("Successfully indexed %d chunks.", len(records))


if __name__ == "__main__":
    build_vector_store()
