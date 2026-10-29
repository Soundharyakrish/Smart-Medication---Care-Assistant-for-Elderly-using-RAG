
import pytest

import rag_pipeline as rag


# ==================================================
# QUESTION VALIDATION
# ==================================================

@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_generate_answer_rejects_empty_question(question):
    with pytest.raises(ValueError):
        rag.generate_answer(question)


def test_generate_answer_rejects_question_over_limit():
    question = "a" * (rag.MAX_QUESTION_CHARS + 1)

    with pytest.raises(ValueError):
        rag.generate_answer(question)


# ==================================================
# EMERGENCY DETECTION
# ==================================================

@pytest.mark.parametrize(
    "question",
    [
        "The person is unconscious",
        "They are not breathing",
        "I have severe chest pain",
        "I think someone overdosed",
        "They are having difficulty breathing",
        "Could this be a stroke?",
        "The person has slurred speech",
        "Their face suddenly started drooping",
        "They are having a seizure",
        "There is severe bleeding",
    ],
)
def test_emergency_questions_are_detected(question):
    assert rag.is_possible_emergency(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "How should medicines be stored?",
        "What is elderly care?",
        "How can I prevent falls?",
        "What is a medication reminder?",
    ],
)
def test_ordinary_questions_are_not_emergencies(question):
    assert rag.is_possible_emergency(question) is False


def test_emergency_response_has_required_fields():
    result = rag.emergency_response()

    assert result["status"] == "emergency"
    assert isinstance(result["answer"], str)
    assert result["answer"].strip()
    assert isinstance(result["sources"], list)


# ==================================================
# FAINTING FIRST AID
# ==================================================

@pytest.mark.parametrize(
    "question",
    [
        "An older adult fainted",
        "Someone has passed out",
        "The person lost consciousness",
    ],
)
def test_fainting_questions_match_first_aid_patterns(question):
    assert rag._matches_any(
        question,
        rag.COMPILED_FAINTING_PATTERNS,
    )


def test_fainting_response_has_required_fields():
    result = rag.fainting_response()

    assert result["status"] == "first_aid"
    assert isinstance(result["answer"], str)
    assert result["answer"].strip()


# ==================================================
# MEDICATION SAFETY
# ==================================================

@pytest.mark.parametrize(
    "question",
    [
        "I took an extra dose",
        "I accidentally took too much medication",
        "I took too many tablets",
        "I took a double dose",
    ],
)
def test_extra_dose_returns_safety_guidance(question):
    result = rag.medication_safety_response(question)

    assert result is not None
    assert result["status"] == "medication_safety"
    assert result["answer"].strip()


@pytest.mark.parametrize(
    "question",
    [
        "I missed a dose",
        "I forgot to take my medicine",
        "I forgot my medication",
    ],
)
def test_missed_dose_returns_safety_guidance(question):
    result = rag.medication_safety_response(question)

    assert result is not None
    assert result["status"] == "medication_safety"
    assert result["answer"].strip()


def test_unrelated_question_has_no_medication_shortcut():
    result = rag.medication_safety_response(
        "How should I help an older adult avoid falls?"
    )

    assert result is None


# ==================================================
# ANSWER CLEANING
# ==================================================

def test_clean_answer_removes_source_labels():
    answer = "Store medicines safely. [Source 1]"

    cleaned = rag.clean_answer(answer)

    assert "[Source 1]" not in cleaned
    assert "Store medicines safely." in cleaned


def test_clean_answer_handles_empty_input():
    assert rag.clean_answer("") == ""
    assert rag.clean_answer(None) == ""


# ==================================================
# GEMINI RESPONSE VALIDATION
# ==================================================

def test_generate_answer_handles_empty_gemini_response(monkeypatch):
    # Use a harmless fake document to avoid loading FAISS.
    monkeypatch.setattr(
        rag,
        "retrieve_documents",
        lambda question: [
            {
                "text": "Store medicines according to their labels.",
                "score": 0.8,
                "source": "test_document.pdf",
                "page": 1,
                "chunk": 0,
            }
        ],
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            class FakeResponse:
                text = ""

            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(
        rag,
        "get_gemini_client",
        lambda: FakeClient(),
    )

    with pytest.raises(RuntimeError, match="empty response"):
        rag.generate_answer(
            "How should medicines be stored safely?"
        )
