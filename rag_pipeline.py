
import json
import logging
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "vector_store"

INDEX_PATH = STORE_DIR / "medical.index"
CHUNKS_PATH = STORE_DIR / "chunks.json"
CONFIG_PATH = STORE_DIR / "config.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
MIN_SIMILARITY = 0.20
MAX_QUESTION_CHARS = 2000

load_dotenv(BASE_DIR / ".env")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_embedding_model = None
_vector_index = None
_document_chunks = None
_gemini_client = None


# ============================================================
# 2. EMERGENCY DETECTION
# ============================================================

EMERGENCY_PATTERNS = [
    # Consciousness and breathing
    r"\bunconscious\b",
    r"\bnot breathing\b",
    r"\bno breathing\b",
    r"\bcan't breathe\b",
    r"\bcannot breathe\b",
    r"\bcan not breathe\b",
    r"\bdifficulty breathing\b",
    r"\btrouble breathing\b",
    r"\bstruggling to breathe\b",
    r"\bsevere shortness of breath\b",
    r"\bunresponsive\b",
    r"\bnot responding\b",

    # Possible heart attack
    r"\bsevere chest pain\b",
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bcrushing chest\b",

    # Possible stroke
    r"\bstroke\b",
    r"\bslurred speech\b",
    r"\bface droop(?:ing|ed|s)?\b",
    r"\bface\b.*\bdroop(?:ing|ed|s)?\b",
    r"\bone side of (?:the )?face\b.*\bweak\b",
    r"\bone side of (?:the )?body\b.*\bweak\b",
    r"\bsudden weakness\b",
    r"\bsudden numbness\b",
    r"\bsudden confusion\b",
    r"\bsudden vision loss\b",

    # Other emergencies
    r"\bseizure\b",
    r"\bconvulsion\b",
    r"\bsevere bleeding\b",
    r"\bbleeding heavily\b",
    r"\bbleeding won't stop\b",
    r"\bchoking\b",
    r"\bsevere allergic reaction\b",
    r"\banaphylaxis\b",
    r"\bthink.*overdose\b",
    r"\bsuspected overdose\b",
    r"\baccidental overdose\b",
    r"\boverdosed\b",
]

COMPILED_EMERGENCY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in EMERGENCY_PATTERNS
]


def _matches_any(question, patterns):
    """Check whether any compiled regex matches the question."""
    if not isinstance(question, str):
        return False
    return any(pattern.search(question) for pattern in patterns)


def is_possible_emergency(question: str) -> bool:
    """Detect questions that may describe a medical emergency."""
    if not isinstance(question, str):
        return False

    # Explicit stroke check ensures common stroke questions are caught.
    if re.search(r"\bstroke\b", question, re.IGNORECASE):
        return True

    return _matches_any(question, COMPILED_EMERGENCY_PATTERNS)


def is_emergency(question: str) -> bool:
    """Backward-compatible emergency detector."""
    return is_possible_emergency(question)


def emergency_response():
    """Return immediate safety guidance without calling Gemini."""
    return {
        "status": "emergency",
        "answer": (
            "This may be a medical emergency. Contact your local emergency "
            "services immediately. In India, call 112 for emergency assistance. "
            "Do not wait for an online response.\n\n"
            "If a stroke is possible, note the time symptoms began and seek "
            "emergency care immediately. Do not give food, drink, or medication "
            "unless instructed by a qualified emergency professional. Follow "
            "the emergency dispatcher's instructions."
        ),
        "sources": [],
    }


# ============================================================
# 3. FAINTING / FIRST AID
# ============================================================

FAINTING_PATTERNS = [
    r"\bfainted\b",
    r"\bfainting\b",
    r"\bpassed out\b",
    r"\blost consciousness\b",
    r"\bloss of consciousness\b",
    r"\bblack(?:ed)? out\b",
]

COMPILED_FAINTING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in FAINTING_PATTERNS
]


def fainting_response():
    return {
        "status": "first_aid",
        "answer": (
            "If someone has fainted, help them lie flat in a safe place and "
            "check whether they are responsive and breathing normally. Do not "
            "make them stand up quickly, and do not give food, drink, or oral "
            "medication while they are unconscious.\n\n"
            "Call emergency services if they are not breathing normally, do "
            "not regain consciousness promptly, are injured, have chest pain, "
            "have difficulty breathing, or show signs of a stroke. Follow the "
            "emergency dispatcher's instructions and seek medical assessment, "
            "especially after a first or unexplained episode."
        ),
        "sources": [],
    }


def _is_fainting_question(question: str) -> bool:
    return _matches_any(question, COMPILED_FAINTING_PATTERNS)


# ============================================================
# 4. MEDICATION SAFETY
# ============================================================

EXTRA_DOSE_PATTERNS = [
    r"\bextra dose\b",
    r"\btook too much\b",
    r"\btaken too much\b",
    r"\btoo many tablets\b",
    r"\btoo many pills\b",
    r"\bdouble dose\b",
    r"\boverdose\b",
    r"\boverdosed\b",
    r"\baccidentally took more\b",
]

MISSED_DOSE_PATTERNS = [
    r"\bmissed (?:a )?dose\b",
    r"\bmiss(?:ed|ing) my dose\b",
    r"\bforgot to take\b",
    r"\bforgot my medication\b",
    r"\bforgot my medicine\b",
    r"\bdidn't take my medicine\b",
    r"\bdid not take my medicine\b",
    r"\bdidn't take my medication\b",
    r"\bdid not take my medication\b",
]

COMPILED_EXTRA_DOSE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in EXTRA_DOSE_PATTERNS
]

COMPILED_MISSED_DOSE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in MISSED_DOSE_PATTERNS
]


def medication_safety_response(question: str):
    """Handle extra-dose and missed-dose questions conservatively."""
    if not isinstance(question, str):
        return None

    if _matches_any(question, COMPILED_EXTRA_DOSE_PATTERNS):
        return {
            "status": "medication_safety",
            "answer": (
                "Taking more medicine than prescribed can be dangerous. "
                "Contact a poison information service, pharmacist, or medical "
                "professional immediately for advice about the medicine and "
                "amount taken. Do not wait for symptoms or try home remedies. "
                "Do not induce vomiting.\n\n"
                "Keep the medicine packaging available and note how much was "
                "taken and when. If the person collapses, has a seizure, has "
                "trouble breathing, or is difficult to wake, contact emergency "
                "services immediately."
            ),
            "sources": [],
        }

    if _matches_any(question, COMPILED_MISSED_DOSE_PATTERNS):
        return {
            "status": "medication_safety",
            "answer": (
                "What to do after a missed dose depends on the specific "
                "medicine and your dosing schedule. Check the patient leaflet "
                "or contact your pharmacist or prescribing clinician.\n\n"
                "Do not take a double dose to make up for a missed dose unless "
                "the medicine's official instructions or a qualified healthcare "
                "professional specifically directs you to do so."
            ),
            "sources": [],
        }

    return None


# ============================================================
# 5. EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


# ============================================================
# 6. FAISS VECTOR STORE
# ============================================================

def load_vector_store():
    global _vector_index, _document_chunks

    if _vector_index is not None and _document_chunks is not None:
        return _vector_index, _document_chunks

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}. "
            "Run ingest_documents.py first."
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Document chunks not found: {CHUNKS_PATH}. "
            "Run ingest_documents.py first."
        )

    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as file:
                config = json.load(file)

            configured_model = config.get("embedding_model")
            if (
                configured_model
                and configured_model != EMBEDDING_MODEL_NAME
            ):
                raise ValueError(
                    "The vector store uses a different embedding model. "
                    "Rebuild it using ingest_documents.py."
                )
        except ValueError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read vector-store config: {exc}"
            ) from exc

    index = faiss.read_index(str(INDEX_PATH))

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if isinstance(chunks, dict):
        if isinstance(chunks.get("chunks"), list):
            chunks = chunks["chunks"]
        elif isinstance(chunks.get("documents"), list):
            chunks = chunks["documents"]
        else:
            raise ValueError(
                "chunks.json must contain a list or a dictionary with "
                "'chunks' or 'documents' as a list."
            )

    if not isinstance(chunks, list):
        raise ValueError("Document chunks must be a JSON list.")

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS has {index.ntotal} vectors but chunks.json has "
            f"{len(chunks)} chunks. Rebuild the vector store."
        )

    model = get_embedding_model()
    dimension = model.get_sentence_embedding_dimension()

    if index.d != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: index={index.d}, "
            f"model={dimension}. Rebuild the vector store."
        )

    _vector_index = index
    _document_chunks = chunks

    logger.info("Loaded vector store with %s chunks.", len(chunks))
    return _vector_index, _document_chunks


# ============================================================
# 7. GEMINI CLIENT
# ============================================================

def get_gemini_client():
    global _gemini_client

    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )

        _gemini_client = genai.Client(api_key=api_key)

    return _gemini_client


# ============================================================
# 8. DOCUMENT RETRIEVAL
# ============================================================

def retrieve_documents(question: str):
    index, chunks = load_vector_store()
    model = get_embedding_model()

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    distances, indices = index.search(query_vector, TOP_K)
    retrieved = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[int(idx)]
        score = float(distance)

        if score < MIN_SIMILARITY:
            continue

        if isinstance(chunk, str):
            text = chunk
            metadata = {}
        elif isinstance(chunk, dict):
            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or ""
            )
            metadata = chunk
        else:
            continue

        if not isinstance(text, str) or not text.strip():
            continue

        retrieved.append(
            {
                "text": text.strip(),
                "score": score,
                "source": metadata.get(
                    "source",
                    metadata.get("filename", "Reference document"),
                ),
                "page": metadata.get("page"),
                "chunk": metadata.get("chunk"),
            }
        )

    return retrieved


# ============================================================
# 9. PROMPT CONSTRUCTION
# ============================================================

def build_prompt(question: str, retrieved_documents: list) -> str:
    if not retrieved_documents:
        return (
            "You are a careful medication and elderly-care assistant. "
            "The available documents do not provide enough evidence to answer.\n\n"
            f"User question: {question}\n\n"
            "Explain that you could not find enough relevant information. "
            "Recommend checking with a pharmacist or qualified healthcare "
            "professional for medicine-specific advice. Do not invent "
            "dosages, diagnoses, or treatment instructions."
        )

    excerpts = []

    for number, document in enumerate(retrieved_documents, start=1):
        excerpts.append(
            f"Reference excerpt {number}:\n{document['text']}"
        )

    context = "\n\n".join(excerpts)

    return f"""
You are Smart Medication & Care Assistant for Elderly, a cautious
health-information assistant for older adults and caregivers.

Answer the user's question using the reference excerpts below.

SAFETY RULES:
- Use the excerpts as the primary evidence.
- Never invent medication dosages, diagnoses, or treatment plans.
- If the excerpts do not provide enough information, say so clearly.
- Never advise doubling a missed dose.
- For overdose or dangerous symptoms, recommend prompt professional help.
- Do not claim to replace a doctor, pharmacist, or emergency service.
- Use plain, friendly, easy-to-understand language.
- Keep the answer concise and easy to read on a laptop screen.
- Do not show source labels, filenames, or a separate sources section.

REFERENCE EXCERPTS:
{context}

USER QUESTION:
{question}

Write the answer directly to the user.
""".strip()


# ============================================================
# 10. ANSWER CLEANING
# ============================================================

def clean_answer(answer) -> str:
    if not isinstance(answer, str) or not answer.strip():
        return ""

    cleaned = answer.strip()

    cleaned = re.sub(
        r"\[\s*source\s*\d+\s*\]",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove standalone source/reference lines.
    cleaned = re.sub(
        r"(?im)^\s*(?:sources?|references?)\s*:\s*.*$",
        "",
        cleaned,
    )

    # Remove a multi-line Sources/References section.
    cleaned = re.sub(
        r"(?ims)^\s*(?:sources?|references?)\s*:\s*\n.*$",
        "",
        cleaned,
    )

    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def clean_answer_text(answer) -> str:
    return clean_answer(answer)


# ============================================================
# 11. MAIN RAG FUNCTION
# ============================================================

def generate_answer(question: str) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")

    question = question.strip()

    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"Your question is too long. Please keep it under "
            f"{MAX_QUESTION_CHARS} characters."
        )

    # 1. Emergency detection must run first.
    if is_possible_emergency(question):
        return emergency_response()

    # 2. Fainting first-aid shortcut.
    if _is_fainting_question(question):
        return fainting_response()

    # 3. Medication safety shortcuts.
    safety_result = medication_safety_response(question)
    if safety_result is not None:
        return safety_result

    # 4. Retrieve evidence.
    retrieved_documents = retrieve_documents(question)

    if not retrieved_documents:
        return {
            "status": "insufficient_evidence",
            "answer": (
                "I couldn't find enough relevant information in the available "
                "care documents to answer that confidently. Please check the "
                "medicine's official leaflet or ask a pharmacist or qualified "
                "healthcare professional for advice."
            ),
            "sources": [],
        }

    # 5. Generate an answer using Gemini.
    prompt = build_prompt(question, retrieved_documents)

    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = clean_answer(getattr(response, "text", "") or "")

        if not answer:
            raise RuntimeError(
                "Gemini returned an empty response. Please try again."
            )

        return {
            "status": "answered",
            "answer": answer,
            "sources": [
                {
                    "source": document.get("source"),
                    "page": document.get("page"),
                    "chunk": document.get("chunk"),
                    "score": document.get("score"),
                }
                for document in retrieved_documents
            ],
        }

    except RuntimeError:
        raise
    except Exception as exc:
        logger.exception("Gemini answer generation failed.")
        raise RuntimeError(
            "I couldn't generate an answer right now. Please try again. "
            "If the issue continues, check the application logs."
        ) from exc
