"""Routes serving static API documentation assets."""


def register_static_routes(app_context):
    """Register root and OpenAPI UI routes."""

    @app_context.app.route("/", methods=["GET"])
    def index():
        return openapi()

    @app_context.app.route("/openapi", methods=["GET"])
    def openapi():
        return app_context.app.send_static_file("index.html")
