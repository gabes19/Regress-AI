"""Optional Google OpenID Connect helpers."""

import hmac
import secrets

from flask import session


def configure_google_oauth(app):
    if not (app.config.get("GOOGLE_CLIENT_ID") and app.config.get("GOOGLE_CLIENT_SECRET")):
        return None
    try:
        from authlib.integrations.flask_client import OAuth
    except ImportError:
        app.logger.warning("Google login disabled because Authlib is not installed.")
        return None
    oauth = OAuth(app)
    oauth.register(
        name="google",
        client_id=app.config["GOOGLE_CLIENT_ID"],
        client_secret=app.config["GOOGLE_CLIENT_SECRET"],
        server_metadata_url=app.config["GOOGLE_DISCOVERY_URL"],
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth.google


def current_user():
    user = session.get("user")
    return user if isinstance(user, dict) else None


def logout_csrf_token():
    token = session.get("logout_csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["logout_csrf_token"] = token
    return token


def validate_logout_csrf(token):
    expected = session.get("logout_csrf_token")
    return bool(expected and token and hmac.compare_digest(expected, token))
