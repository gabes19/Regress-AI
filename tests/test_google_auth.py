import app as app_module

from regressionlab.services.gpu_usage import initialize_gpu_database


class FakeGoogle:
    def authorize_redirect(self, redirect_uri):
        return f"redirect:{redirect_uri}", 302

    def authorize_access_token(self):
        return {"userinfo": {
            "sub": "google-1", "email": "user@example.com",
            "email_verified": True, "name": "Example User",
        }}


def test_google_callback_stores_minimal_session_identity(tmp_path, monkeypatch):
    database = tmp_path / "auth.sqlite"
    initialize_gpu_database(database)
    monkeypatch.setitem(app_module.app.config, "GPU_USAGE_DATABASE", database)
    monkeypatch.setitem(app_module.app.extensions, "google_oauth", FakeGoogle())
    client = app_module.app.test_client()

    response = client.get("/auth/google/callback")
    assert response.status_code == 302
    with client.session_transaction() as session:
        assert session["user"]["email"] == "user@example.com"
        assert "token" not in session
        assert "google_sub" not in session["user"]


def test_logout_requires_csrf_and_clears_session(tmp_path, monkeypatch):
    database = tmp_path / "auth.sqlite"
    initialize_gpu_database(database)
    monkeypatch.setitem(app_module.app.config, "GPU_USAGE_DATABASE", database)
    client = app_module.app.test_client()
    with client.session_transaction() as session:
        session["user"] = {"id": 1, "email": "user@example.com"}
        session["logout_csrf_token"] = "valid-token"

    assert client.post("/logout", data={"csrf_token": "wrong"}).status_code == 400
    response = client.post("/logout", data={"csrf_token": "valid-token"})
    assert response.status_code == 302
    with client.session_transaction() as session:
        assert "user" not in session


def test_missing_google_configuration_is_recoverable(monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "google_oauth", None)
    response = app_module.app.test_client().get("/login")
    assert response.status_code == 503
