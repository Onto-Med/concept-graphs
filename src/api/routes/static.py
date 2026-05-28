"""Routes serving static API documentation assets."""


def register_static_routes(main_objects):
    """Register root and OpenAPI UI routes."""

    @main_objects.app.route("/", methods=["GET"])
    def index():
        return openapi()

    @main_objects.app.route("/openapi", methods=["GET"])
    def openapi():
        return main_objects.app.send_static_file("index.html")
