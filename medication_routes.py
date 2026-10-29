# medication_routes.py

import json
import re
from datetime import date

from flask import Blueprint, jsonify, request

import medication_store as store


medication_bp = Blueprint("medication", __name__)


# =========================================================
# COMMON HELPERS
# =========================================================

def error(message, status=400):
    return jsonify({"error": message}), status


def valid_iso_date(value, field_name):
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        )

    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        )

    if parsed.isoformat() != value:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        )

    return parsed


def get_json_object():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    return data


def parse_schedule(data):
    name = data.get("medicine_name", "")
    dosage = data.get("dosage", "")
    instructions = data.get("instructions", "")

    if not isinstance(name, str):
        raise ValueError("Medicine name must be text.")

    if not isinstance(dosage, str):
        raise ValueError("Dosage must be text.")

    if not isinstance(instructions, str):
        raise ValueError("Instructions must be text.")

    name = name.strip()
    dosage = dosage.strip()
    instructions = instructions.strip()

    if not name or len(name) > 150:
        raise ValueError(
            "Enter a medicine name (maximum 150 characters)."
        )

    if not dosage or len(dosage) > 100:
        raise ValueError(
            "Enter the prescribed dosage (maximum 100 characters)."
        )

    if len(instructions) > 1000:
        raise ValueError(
            "Instructions must be 1000 characters or fewer."
        )

    times = data.get("times")

    if not isinstance(times, list) or not times:
        raise ValueError("Add at least one scheduled time.")

    if len(times) > 12:
        raise ValueError(
            "A maximum of 12 daily times is allowed."
        )

    clean_times = []

    for item in times:
        if (
            not isinstance(item, str)
            or not re.fullmatch(
                r"(?:[01]\d|2[0-3]):[0-5]\d",
                item
            )
        ):
            raise ValueError(
                "Times must use 24-hour HH:MM format."
            )

        clean_times.append(item)

    # Remove duplicate times and keep a predictable order.
    clean_times = sorted(set(clean_times))

    start_date = data.get(
        "start_date",
        date.today().isoformat()
    )

    end_date = data.get("end_date", "")

    if end_date is None:
        end_date = ""

    if not isinstance(end_date, str):
        raise ValueError(
            "End date must use YYYY-MM-DD format."
        )

    end_date = end_date.strip()

    start = valid_iso_date(start_date, "Start date")

    if end_date:
        end = valid_iso_date(end_date, "End date")

        if end < start:
            raise ValueError(
                "End date cannot be before start date."
            )

    return {
        "medicine_name": name,
        "dosage": dosage,
        "instructions": instructions,
        "times_json": json.dumps(clean_times),
        "start_date": start.isoformat(),
        "end_date": end_date
    }


# =========================================================
# LIST MEDICATIONS
# =========================================================

@medication_bp.get(
    "/api/elderly/<int:elderly_id>/medications"
)
def list_medications(elderly_id):
    if not store.profile_exists(elderly_id):
        return error("Elderly profile not found.", 404)

    medicines = store.get_medications(elderly_id)

    for medicine in medicines:
        try:
            times = json.loads(medicine.pop("times_json"))
        except (json.JSONDecodeError, TypeError):
            return error(
                "A saved medication has invalid schedule data. "
                "Review that medication before continuing.",
                500
            )

        if not isinstance(times, list):
            return error(
                "A saved medication has invalid schedule data.",
                500
            )

        medicine["times"] = times

    return jsonify({"medications": medicines})


# =========================================================
# CREATE MEDICATION
# =========================================================

@medication_bp.post(
    "/api/elderly/<int:elderly_id>/medications"
)
def add_medication(elderly_id):
    if not store.profile_exists(elderly_id):
        return error("Elderly profile not found.", 404)

    try:
        data = get_json_object()
        medicine = parse_schedule(data)

        medication_id = store.create_medication(
            elderly_id=elderly_id,
            **medicine
        )

    except ValueError as exc:
        return error(str(exc))

    return jsonify({
        "message": "Medication schedule created.",
        "medication_id": medication_id
    }), 201


# =========================================================
# UPDATE MEDICATION
# =========================================================

@medication_bp.put(
    "/api/elderly/<int:elderly_id>/medications/<int:medication_id>"
)
def edit_medication(elderly_id, medication_id):
    if not store.profile_exists(elderly_id):
        return error("Elderly profile not found.", 404)

    existing = store.get_medication(
        medication_id,
        elderly_id
    )

    if not existing or not existing["active"]:
        return error("Active medication not found.", 404)

    try:
        data = get_json_object()
        medicine = parse_schedule(data)

        updated = store.update_medication(
            medication_id=medication_id,
            elderly_id=elderly_id,
            **medicine
        )

    except ValueError as exc:
        return error(str(exc))

    if not updated:
        return error(
            "Medication could not be updated.",
            404
        )

    return jsonify({
        "message": "Medication updated."
    })


# =========================================================
# DEACTIVATE MEDICATION
# =========================================================

@medication_bp.delete(
    "/api/elderly/<int:elderly_id>/medications/<int:medication_id>"
)
def delete_medication(elderly_id, medication_id):
    if not store.profile_exists(elderly_id):
        return error("Elderly profile not found.", 404)

    existing = store.get_medication(
        medication_id,
        elderly_id
    )

    if not existing or not existing["active"]:
        return error("Active medication not found.", 404)

    store.deactivate_medication(
        medication_id,
        elderly_id
    )

    return jsonify({
        "message": "Medication schedule deactivated."
    })


# =========================================================
# LIST DAILY DOSES
# =========================================================

@medication_bp.get(
    "/api/elderly/<int:elderly_id>/doses"
)
def list_daily_doses(elderly_id):
    if not store.profile_exists(elderly_id):
        return error("Elderly profile not found.", 404)

    requested_date = request.args.get(
        "date",
        date.today().isoformat()
    )

    try:
        valid_iso_date(requested_date, "Date")

        doses = store.get_daily_doses(
            elderly_id,
            requested_date
        )

    except ValueError as exc:
        return error(str(exc))

    return jsonify({
        "date": requested_date,
        "doses": doses
    })


# =========================================================
# UPDATE DOSE STATUS
# =========================================================

@medication_bp.patch(
    "/api/elderly/<int:elderly_id>/doses/<int:dose_id>"
)
def update_dose(elderly_id, dose_id):
    if not store.profile_exists(elderly_id):
        return error("Elderly profile not found.", 404)

    try:
        data = get_json_object()
    except ValueError as exc:
        return error(str(exc))

    status = data.get("status")

    if status not in ("Taken", "Skipped", "Pending"):
        return error(
            "Status must be Taken, Skipped, or Pending."
        )

    try:
        updated = store.update_dose_status(
            dose_id,
            elderly_id,
            status
        )
    except ValueError as exc:
        return error(str(exc))

    if not updated:
        return error("Dose record not found.", 404)

    return jsonify({
        "message": "Dose status updated.",
        "status": status
    })


# =========================================================
# DOSE HISTORY
# =========================================================

@medication_bp.get(
    "/api/elderly/<int:elderly_id>/dose-history"
)
def dose_history(elderly_id):
    if not store.profile_exists(elderly_id):
        return error("Elderly profile not found.", 404)

    try:
        limit = int(request.args.get("limit", 100))
    except (TypeError, ValueError):
        return error("Limit must be a number.")

    limit = max(1, min(limit, 200))

    return jsonify({
        "history": store.get_dose_history(
            elderly_id,
            limit
        )
    })