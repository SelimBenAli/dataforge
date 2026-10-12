"""Local Flask dashboard. Imported only when the dashboard is requested."""
from flask import Flask
from pathlib import Path
import secrets

MAX_UPLOAD_SIZE = 25 * 1024 * 1024


def create_app(test_config=None):
    """Create the local dashboard application without starting a server."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(SECRET_KEY=secrets.token_hex(32), MAX_CONTENT_LENGTH=MAX_UPLOAD_SIZE + 64 * 1024,
                      MAX_UPLOAD_SIZE=MAX_UPLOAD_SIZE,
                      UPLOAD_FOLDER=str(Path.cwd() / ".dataforge" / "uploads"))
    if test_config: app.config.update(test_config)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    from .routes import bp
    app.register_blueprint(bp)
    @app.errorhandler(413)
    def too_large(_error): return {"error": "File is larger than the 25 MB upload limit."}, 413
    return app
