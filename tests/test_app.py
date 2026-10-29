
import app as flask_app


# ==================================================
# TEST CONFIGURATION
# ==================================================

def setup_function():
    flask_app.app.config["TESTING"] = True


# ==================================================
# HEALTH CHECK
# ==================================================

def test_health_endpoint():
    client = flask_app.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json()["status"] == "ok"


# ==================================================
# HOME PAGE
# ==================================================

def test_home_page_loads():
    client = flask_app.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Smart Medication" in response.data


# ==================================================
# ASK ENDPOINT
# ==================================================

def test_ask_returns_json_answer(monkeypatch):
    def fake_generate_answer(question):
        assert question == "How should medicines be stored?"

        return {
            "status": "answered",
            "answer": "Store medicines according to the label.",
            "sources": [],
        }

    monkeypatch.setattr(
        flask_app,
        "generate_answer",
        fake_generate_answer,
    )

    client = flask_app.app.test_client()

    response = client.post(
        "/ask",
        json={"question": "How should medicines be stored?"},
    )

    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()

    assert data["status"] == "answered"
    assert "Store medicines" in data["answer"]


def test_ask_rejects_empty_question():
    client = flask_app.app.test_client()

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.is_json


def test_ask_rejects_missing_question():
    client = flask_app.app.test_client()

    response = client.post("/ask", json={})

    assert response.status_code == 400
    assert response.is_json


def test_ask_rejects_non_json_request():
    client = flask_app.app.test_client()

    response = client.post(
        "/ask",
        data="question=hello",
        content_type="application/x-www-form-urlencoded",
    )

    assert response.status_code == 400
    assert response.is_json


def test_ask_rejects_long_question():
    client = flask_app.app.test_client()

    response = client.post(
        "/ask",
        json={"question": "x" * 2001},
    )

    assert response.status_code == 400
    assert response.is_json


def test_ask_returns_json_when_pipeline_fails(monkeypatch):
    def fake_generate_answer(question):
        raise RuntimeError("Simulated pipeline failure")

    monkeypatch.setattr(
        flask_app,
        "generate_answer",
        fake_generate_answer,
    )

    client = flask_app.app.test_client()

    response = client.post(
        "/ask",
        json={"question": "How should medicines be stored?"},
    )

    assert response.status_code == 500
    assert response.is_json

    data = response.get_json()

    assert data["status"] == "error"
    assert data["answer"]


def test_ask_handles_pipeline_returning_empty_answer(monkeypatch):
    def fake_generate_answer(question):
        return {
            "status": "answered",
            "answer": "",
            "sources": [],
        }

    monkeypatch.setattr(
        flask_app,
        "generate_answer",
        fake_generate_answer,
    )

    client = flask_app.app.test_client()

    response = client.post(
        "/ask",
        json={"question": "How should medicines be stored?"},
    )

    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()

    assert data["answer"].strip()
    assert data["status"] == "insufficient_evidence"
