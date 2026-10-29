
import re

from flask import Blueprint, jsonify, render_template, request

import caregiver_store as store

caregiver_bp = Blueprint("caregivers", __name__)

ALLOWED_ROLES = {"primary", "backup", "doctor"}
MAX_NAME_LENGTH = 100
MAX_PHONE_LENGTH = 25
MAX_NOTES_LENGTH = 1000


def error_response(message, status=400):
    return jsonify({"status": "error", "message": message}), status


def parse_id(value):
    try:
        number = int(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def validate_text(value, label, required=False, maximum=MAX_NOTES_LENGTH):
    if value is None:
        value = ""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")

    value = value.strip()

    if required and not value:
        raise ValueError(f"{label} is required.")

    if len(value) > maximum:
        raise ValueError(f"{label} must be at most {maximum} characters.")

    return value


def validate_age(value):
    if value in (None, ""):
        return None

    try:
        age = int(value)
    except (TypeError, ValueError):
        raise ValueError("Age must be a whole number.")

    if not 0 <= age <= 125:
        raise ValueError("Age must be between 0 and 125.")

    return age


def read_profile_payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Send the profile details as JSON.")

    return {
        "name": validate_text(
            data.get("name"), "Name", required=True,
            maximum=MAX_NAME_LENGTH,
        ),
        "age": validate_age(data.get("age")),
        "notes": validate_text(data.get("notes"), "Notes"),
    }


def read_contact_payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Send the contact details as JSON.")

    role = validate_text(
        data.get("role"), "Contact type", required=True, maximum=20
    ).lower()

    if role not in ALLOWED_ROLES:
        raise ValueError("Contact type must be primary, backup, or doctor.")

    name = validate_text(
        data.get("name"), "Contact name", required=True,
        maximum=MAX_NAME_LENGTH,
    )
    phone = validate_text(
        data.get("phone"), "Phone number", required=True,
        maximum=MAX_PHONE_LENGTH,
    )

    # Allow common international phone-number formatting.
    if not re.fullmatch(r"[0-9+()\-\s.]+", phone):
        raise ValueError("Enter a valid phone number.")

    digits = re.sub(r"\D", "", phone)
    if not 7 <= len(digits) <= 15:
        raise ValueError("Phone number must contain 7 to 15 digits.")

    return {
        "role": role,
        "name": name,
        "phone": phone,
        "relationship": validate_text(
            data.get("relationship"), "Relationship", maximum=100
        ),
        "organization": validate_text(
            data.get("organization"), "Clinic or organization", maximum=150
        ),
        "notes": validate_text(data.get("notes"), "Notes"),
    }


def profile_or_404(elderly_id):
    profile = store.get_profile(elderly_id)
    if profile is None:
        return None
    return profile


@caregiver_bp.route("/caregivers", methods=["GET"])
def caregiver_page():
    return render_template("caregivers.html")


@caregiver_bp.route("/api/elderly", methods=["GET"])
def api_list_profiles():
    return jsonify({"profiles": store.list_profiles()})


@caregiver_bp.route("/api/elderly", methods=["POST"])
def api_create_profile():
    try:
        payload = read_profile_payload()
        profile_id = store.create_profile(**payload)
        return jsonify({
            "status": "success",
            "profile": store.get_profile(profile_id),
        }), 201
    except ValueError as exc:
        return error_response(str(exc))


@caregiver_bp.route("/api/elderly/<int:elderly_id>", methods=["GET"])
def api_get_profile(elderly_id):
    profile = profile_or_404(elderly_id)
    if profile is None:
        return error_response("Elderly profile not found.", 404)

    profile["contacts"] = store.list_contacts(elderly_id)
    return jsonify({"profile": profile})


@caregiver_bp.route("/api/elderly/<int:elderly_id>", methods=["PUT"])
def api_update_profile(elderly_id):
    if profile_or_404(elderly_id) is None:
        return error_response("Elderly profile not found.", 404)

    try:
        payload = read_profile_payload()
        store.update_profile(elderly_id, **payload)
        return jsonify({
            "status": "success",
            "profile": store.get_profile(elderly_id),
        })
    except ValueError as exc:
        return error_response(str(exc))


@caregiver_bp.route("/api/elderly/<int:elderly_id>", methods=["DELETE"])
def api_delete_profile(elderly_id):
    if not store.delete_profile(elderly_id):
        return error_response("Elderly profile not found.", 404)

    return jsonify({"status": "success"})


@caregiver_bp.route(
    "/api/elderly/<int:elderly_id>/contacts", methods=["GET"]
)
def api_list_contacts(elderly_id):
    if profile_or_404(elderly_id) is None:
        return error_response("Elderly profile not found.", 404)

    return jsonify({"contacts": store.list_contacts(elderly_id)})


@caregiver_bp.route(
    "/api/elderly/<int:elderly_id>/contacts", methods=["POST"]
)
def api_create_contact(elderly_id):
    if profile_or_404(elderly_id) is None:
        return error_response("Elderly profile not found.", 404)

    try:
        payload = read_contact_payload()
        contact_id = store.create_contact(elderly_id, **payload)
        return jsonify({
            "status": "success",
            "contact": store.get_contact(contact_id, elderly_id),
        }), 201
    except ValueError as exc:
        return error_response(str(exc))


@caregiver_bp.route(
    "/api/elderly/<int:elderly_id>/contacts/<int:contact_id>",
    methods=["PUT"],
)
def api_update_contact(elderly_id, contact_id):
    if profile_or_404(elderly_id) is None:
        return error_response("Elderly profile not found.", 404)

    if store.get_contact(contact_id, elderly_id) is None:
        return error_response("Contact not found for this profile.", 404)

    try:
        payload = read_contact_payload()
        store.update_contact(contact_id, elderly_id, **payload)
        return jsonify({
            "status": "success",
            "contact": store.get_contact(contact_id, elderly_id),
        })
    except ValueError as exc:
        return error_response(str(exc))


@caregiver_bp.route(
    "/api/elderly/<int:elderly_id>/contacts/<int:contact_id>",
    methods=["DELETE"],
)
def api_delete_contact(elderly_id, contact_id):
    if not store.delete_contact(contact_id, elderly_id):
        return error_response("Contact not found for this profile.", 404)

    return jsonify({"status": "success"})
