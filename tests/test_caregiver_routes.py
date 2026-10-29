
import pytest

import app as flask_app
import caregiver_store as store


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Use a temporary database so tests cannot change real contact data.
    test_db = tmp_path / "test_caregivers.db"
    monkeypatch.setattr(store, "DATABASE_PATH", test_db)
    store.init_database()

    flask_app.app.config["TESTING"] = True

    with flask_app.app.test_client() as test_client:
        yield test_client


def create_profile(client, name="Test Elder"):
    response = client.post(
        "/api/elderly",
        json={"name": name, "age": 75, "notes": ""},
    )
    assert response.status_code == 201
    return response.get_json()["profile"]["id"]


def test_caregiver_page_loads(client):
    response = client.get("/caregivers")
    assert response.status_code == 200
    assert b"Caregiver Contacts" in response.data


def test_create_and_list_profiles(client):
    profile_id = create_profile(client, "Test Person")

    response = client.get("/api/elderly")
    assert response.status_code == 200
    assert any(
        profile["id"] == profile_id
        for profile in response.get_json()["profiles"]
    )


def test_contact_belongs_to_selected_profile(client):
    first_id = create_profile(client, "Person One")
    second_id = create_profile(client, "Person Two")

    response = client.post(
        f"/api/elderly/{first_id}/contacts",
        json={
            "role": "primary",
            "name": "Test Caregiver",
            "phone": "+91 9876543210",
        },
    )
    assert response.status_code == 201

    first_contacts = client.get(
        f"/api/elderly/{first_id}/contacts"
    ).get_json()["contacts"]

    second_contacts = client.get(
        f"/api/elderly/{second_id}/contacts"
    ).get_json()["contacts"]

    assert len(first_contacts) == 1
    assert first_contacts[0]["name"] == "Test Caregiver"
    assert second_contacts == []


def test_invalid_contact_role_is_rejected(client):
    profile_id = create_profile(client)

    response = client.post(
        f"/api/elderly/{profile_id}/contacts",
        json={
            "role": "unknown",
            "name": "Test Contact",
            "phone": "9876543210",
        },
    )
    assert response.status_code == 400


def test_invalid_phone_is_rejected(client):
    profile_id = create_profile(client)

    response = client.post(
        f"/api/elderly/{profile_id}/contacts",
        json={
            "role": "primary",
            "name": "Test Contact",
            "phone": "abc",
        },
    )
    assert response.status_code == 400


def test_contact_cannot_be_accessed_under_another_profile(client):
    first_id = create_profile(client, "Person One")
    second_id = create_profile(client, "Person Two")

    response = client.post(
        f"/api/elderly/{first_id}/contacts",
        json={
            "role": "backup",
            "name": "Backup Contact",
            "phone": "9876543210",
        },
    )
    contact_id = response.get_json()["contact"]["id"]

    response = client.delete(
        f"/api/elderly/{second_id}/contacts/{contact_id}"
    )
    assert response.status_code == 404

    contacts = client.get(
        f"/api/elderly/{first_id}/contacts"
    ).get_json()["contacts"]
    assert len(contacts) == 1


def test_delete_profile_removes_its_contacts(client):
    profile_id = create_profile(client)

    client.post(
        f"/api/elderly/{profile_id}/contacts",
        json={
            "role": "doctor",
            "name": "Test Doctor",
            "phone": "9876543210",
        },
    )

    response = client.delete(f"/api/elderly/{profile_id}")
    assert response.status_code == 200

    # The profile no longer exists.
    response = client.get(f"/api/elderly/{profile_id}")
    assert response.status_code == 404
