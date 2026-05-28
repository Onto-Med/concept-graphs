"""Routes serving static API documentation assets."""

from flask import Blueprint


def create_static_blueprint(app):
    """Create the blueprint for root and OpenAPI UI routes."""
    blueprint = Blueprint("static_routes", __name__)

    @blueprint.route("/", methods=["GET"])
    def index():
        return openapi()

    @blueprint.route("/openapi", methods=["GET"])
    def openapi():
        return app.send_static_file("index.html")

    return blueprint
