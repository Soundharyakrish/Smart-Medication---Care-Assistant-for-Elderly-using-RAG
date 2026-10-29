
import os

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from rag_pipeline import generate_answer
from caregiver_routes import caregiver_bp
from medication_routes import medication_bp


# --------------------------------------------------
# ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


# --------------------------------------------------
# REGISTER BLUEPRINTS
# --------------------------------------------------

app.register_blueprint(caregiver_bp)
app.register_blueprint(medication_bp)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Display the main Smart Medication & Care Assistant page."""
    return render_template("index.html")


# --------------------------------------------------
# ASK THE RAG CARE ASSISTANT
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask():
    """Process a question using the existing RAG pipeline."""

    try:
        # Accept JSON requests from the frontend and API clients.
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            data = {}

        question = data.get("question", "")

        if not isinstance(question, str):
            question = ""

        question = question.strip()

        # Validate empty questions.
        if not question:
            return jsonify({
                "status": "error",
                "answer": "Please enter a question.",
                "sources": [],
                "error": "Question cannot be empty."
            }), 400

        # Validate excessively long questions.
        if len(question) > 2000:
            return jsonify({
                "status": "error",
                "answer": (
                    "Your question is too long. "
                    "Please shorten it and try again."
                ),
                "sources": [],
                "error": "Question exceeds the maximum length of 2000 characters."
            }), 400

        # Call the existing RAG pipeline.
        result = generate_answer(question)

        # Extract the response while supporting dictionary and string results.
        if isinstance(result, dict):
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            pipeline_status = result.get("status")

        elif isinstance(result, str):
            answer = result
            sources = []
            pipeline_status = None

        else:
            answer = ""
            sources = []
            pipeline_status = None

        # Ensure answer is a string.
        if not isinstance(answer, str):
            answer = ""

        answer = answer.strip()

        # Ensure sources are JSON-friendly and consistently represented.
        if not isinstance(sources, list):
            sources = []

        # An empty answer must not be reported as successfully answered.
        if not answer:
            return jsonify({
                "status": "insufficient_evidence",
                "answer": (
                    "I couldn't find enough relevant information "
                    "to answer your question. Please consult a "
                    "healthcare professional for further guidance."
                ),
                "sources": []
            }), 200

        # Preserve the status supplied by the RAG pipeline when available.
        if pipeline_status in (
            "answered",
            "insufficient_evidence",
            "error"
        ):
            response_status = pipeline_status
        else:
            response_status = "answered"

        return jsonify({
            "status": response_status,
            "answer": answer,
            "sources": sources
        }), 200

    except ValueError as exc:
        app.logger.warning("Invalid question or pipeline input: %s", exc)

        return jsonify({
            "status": "error",
            "answer": str(exc) or "The question could not be processed.",
            "sources": [],
            "error": str(exc)
        }), 400

    except Exception as exc:
        # Keep technical details in the server log rather than displaying
        # exception details to the user.
        app.logger.exception("The /ask endpoint failed.")

        return jsonify({
            "status": "error",
            "answer": (
                "Sorry, I couldn't process your question right now. "
                "Please try again later."
            ),
            "sources": [],
            "error": "The RAG pipeline failed."
        }), 500


# --------------------------------------------------
# CAREGIVER DASHBOARD
# --------------------------------------------------

@app.route("/caregivers", methods=["GET"])
def caregivers_page():
    """Display the caregiver dashboard."""
    return render_template("caregivers.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Check whether the Flask application is running."""
    return jsonify({
        "status": "ok",
        "message": "Smart Medication & Care Assistant is running."
    }), 200


# --------------------------------------------------
# REGISTERED ROUTES
# --------------------------------------------------

@app.route("/routes", methods=["GET"])
def list_routes():
    """List registered application routes for development checks."""
    routes = []

    for rule in app.url_map.iter_rules():
        methods = sorted(
            method
            for method in rule.methods
            if method not in ("HEAD", "OPTIONS")
        )

        routes.append({
            "endpoint": rule.endpoint,
            "methods": methods,
            "path": str(rule)
        })

    return jsonify({
        "status": "ok",
        "routes": routes
    }), 200


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=debug_mode
    )
